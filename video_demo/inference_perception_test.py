import os
import re
import math
import json
import traceback
import io
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_video(video_path, strategy='chat'):
    bridge.set_bridge('torch')
    with open(video_path, 'rb') as f:
        mp4_stream = f.read()
    num_frames = 24

    if mp4_stream is not None:
        decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))
    else:
        decord_vr = VideoReader(video_path, ctx=cpu(0))
    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data

class PerceptionTestMCQADataset(Dataset):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_list, stragegy):
        self.data_list = data_list
        self.stragegy = stragegy

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_name = line['metadata']['video_id']
        mc_questions = line['mc_question']

        # for fmt in self.video_formats:  # Added this line
        fmt = '.mp4'  # custom, we only need mp4
        temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
        if os.path.exists(temp_path):
            video_path = temp_path
            video_data = load_video(video_path, self.stragegy)
        else:
            print("warning, video not found", temp_path)
            return None

        instructs = []
        qids = []
        ops = []
        ans = []
        for q in mc_questions:
            question = q['question']
            qid = q['id']
            options = q['options']
            instruct = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'

            instructs.append(instruct)
            qids.append(qid)
            ops.append(options)
            ans.append(q['answer_id'])

        return {
            'video': video_data,
            'video_id': video_name,
            'instructs': instructs,
            'question_ids': qids,
            'options': ops,
            'answer': ans,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    v_id = [x['video_id'] for x in batch]
    ins = [x['instructs'] for x in batch]
    q_ids = [x['question_ids'] for x in batch]
    ops = [x['options'] for x in batch]
    ans = [x['answer'] for x in batch]
    vid = torch.stack(vid, dim=0)
    return vid, v_id, ins, q_ids, ops, ans


def run_inference(args):
    # parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    # parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--answer-file',
    args.video_folder = "/home/jim/Documents/Projects/perception_test/baselines/data/videos"
    args.question_file = "/home/jim/Documents/Projects/perception_test/baselines/data/mc_question_train.json"
    args.answer_file = "./answer_file.json"
    MODEL_PATH = "THUDM/cogvlm2-video-llama3-chat"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        # padding_side="left"
    )

    if torch.cuda.is_available() and torch.cuda.get_device_properties(
            0).total_memory < 48 * 1024 ** 3 and not args.quant:
        print("GPU memory is less than 48GB. Please use cli_demo_multi_gpus.py or pass `--quant 4` or `--quant 8`.")
        exit()

    # Load the model
    if args.quant == 4:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=TORCH_TYPE,
            ),
            low_cpu_mem_usage=True
        ).eval()
    elif args.quant == 8:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=TORCH_TYPE,
            ),
            low_cpu_mem_usage=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True
        ).eval().to(DEVICE)

    strategy = 'base' if 'cogvlm2-video-llama3-base' in MODEL_PATH else 'chat'
    print(f"using with {strategy} model")

    questions = json.load(open(args.question_file, "r"))
    questions = list(questions.values())
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = PerceptionTestMCQADataset(questions, strategy)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_fn)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    MAX_ITER_SIZE = 250

    # Iterate over each sample in the ground truth file
    for i, (video_tensor, video_id, instructs, question_ids, options, true_ans) in enumerate(tqdm(dataloader)):

        # reduce batch dimension
        video_tensor = video_tensor[0]
        video_id = video_id[0]
        instructs = instructs[0]
        question_ids = question_ids[0]
        options = options[0]
        true_ans = true_ans[0]

        qas = []
        history = []
        for idx, instruct in enumerate(instructs):
            letters = ['(A)', '(B)', '(C)']
            question_id = question_ids[idx]
            _options = options[idx]
            q_answer = true_ans[idx]

            # make it such that blank is fed instead of original video tensor:
            blank_video_tensor = torch.zeros_like(video_tensor)
            video_tensor = blank_video_tensor

            inputs = model.build_conversation_input_ids(
                tokenizer=tokenizer,
                query=instruct,
                images=[video_tensor],
                history=history,
                template_version=strategy
            )
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(DEVICE),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(DEVICE),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(DEVICE),
                'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
            }
            gen_kwargs = {
                "max_new_tokens": 2048,
                "pad_token_id": 128002,
                "top_k": 1,
                "do_sample": True,
                "top_p": 0.1,
                "temperature": 0.1,
            }
            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)
                output = output[:, inputs['input_ids'].shape[1]:]
                output = tokenizer.decode(output[0], skip_special_tokens=True)
                # print("\nCogVLM2-Video:", output)

            output = output.replace('answer', '')
            output = output.replace('Answer', '')
            pred_answer = re.findall('\(*[A-C]\)*', output)
            try:
                assert len(
                    pred_answer) >= 1, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(
                    video_id, instruct, output)
                pred_answer = pred_answer[0].strip()
                # if not pred_answer.startswith('('):
                pred_answer = pred_answer.strip('()')
                pred_answer = f'({pred_answer})'
                pred_idx = letters.index(pred_answer)
            except:
                traceback.print_exc()
                tmp_options = [x.lower() for x in _options]
                if output.lower() in tmp_options:
                    tmp_options = [x.lower() for x in _options]
                    pred_idx = tmp_options.index(output.lower())
                else:
                    pred_idx = 2

            qas.append({'id': question_id, 'question': instruct, 'answer_id': pred_idx, 'answer': _options[pred_idx],
                        'answer_text': output, 'true_answer': q_answer, "correct": pred_idx == q_answer})

        ans_file.write('\"{}\": {},\n'.format(video_id, json.dumps(qas)))

        if i > min(MAX_ITER_SIZE, len(dataloader)):
            print("max eval size reached, exiting...")
            break
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    # parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=8)
    args = parser.parse_args()

    run_inference(args)
