# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto


class BatchRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        padded_step_retrieve_entropy = data.batch["padded_step_retrieve_entropy"]
        padded_step_retrieve_token_lengths = data.batch["padded_step_retrieve_token_lengths"]
        padded_step_mask = data.batch["padded_step_mask"]

        scores = self.verify(data)
        rewards = []
        already_printed = {}

        retrieve_gates = torch.zeros(len(data), dtype=torch.float32, device=prompt_ids.device)
        retrieve_rewards = torch.zeros(len(data), dtype=torch.float32, device=prompt_ids.device)
        retrieve_token_ratios = torch.zeros(len(data), dtype=torch.float32, device=prompt_ids.device)

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            # ======= 计算retrieve reward =======
            retrieve_entropy = padded_step_retrieve_entropy[i]
            retrieve_mask = padded_step_mask[i]
            retrieve_token_length = padded_step_retrieve_token_lengths[i]
            # find retrieve steps
            retrieve_position = (retrieve_entropy == -1) & (retrieve_mask == 1)
            retrieve_idx = retrieve_position.nonzero(as_tuple=False).squeeze(-1)
            # compute retrieve drops and record h_pre.
            # If retrieved in the given rollout, compute rollout-based H_drop, H_pre, and token ratios
            if retrieve_idx.numel() > 0:
                gate_list, H_pre_list = [], []
                for each_retrieve_step in retrieve_idx:
                    # H_pre: nearest entropy >=0 before retrieve_step
                    pre_entropy = retrieve_entropy[:each_retrieve_step]
                    pre_mask = (pre_entropy >= 0)
                    # H_post: nearest entropy >=0 after retrieve_step
                    post_entropy = retrieve_entropy[each_retrieve_step + 1:]
                    post_mask = (post_entropy >= 0)
                    if not pre_mask.any() or not post_mask.any():
                        continue
                    H_pre = pre_entropy[pre_mask][-1]
                    H_post = post_entropy[post_mask][0]
                    H_drop = -(H_post - H_pre) / (H_pre + 1e-6)
                    gate = torch.sigmoid((H_drop - 0.0) / 0.5)
                    gate_list.append(gate)
                    H_pre_list.append(H_pre)
                # Mean gate and H_pre for each rollout
                if len(gate_list) > 0:
                    retrieve_gates[i] = torch.stack(gate_list).mean()
                if len(H_pre_list) > 0:
                    retrieve_rewards[i] = torch.stack(H_pre_list).mean()
                # Retrieve ratios
                # 每个 rollout 的 retrieve token 总长度 (bs,)
                retrieve_token = (retrieve_token_length * retrieve_mask).sum(dim=-1)
                # compute drop for each retrieve steps
                retrieve_token_ratios[i] = retrieve_token * 1.0 / max(length, 1.0)

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)
        # data.batch["retrieve_gates"] = retrieve_gates
        # data.batch["retrieve_rewards"] = retrieve_rewards
        # data.batch["retrieve_token_ratios"] = retrieve_token_ratios

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info,
                    "retrieve_gates": retrieve_gates, "retrieve_rewards": retrieve_rewards, "retrieve_token_ratios": retrieve_token_ratios}
        else:
            return reward_tensor
