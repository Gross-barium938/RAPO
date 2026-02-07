# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from verl.utils.reward_score import default_compute_score
import time
import numpy as np

class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # ===== retrieve related tensors (batch-level) =====
        padded_step_retrieve_entropy = data.batch["padded_step_retrieve_entropy"]
        padded_step_retrieve_token_lengths = data.batch["padded_step_retrieve_token_lengths"]
        padded_step_mask = data.batch["padded_step_mask"]

        retrieve_gates = torch.zeros(len(data), dtype=torch.float32, device=padded_step_retrieve_entropy.device)
        retrieve_rewards = torch.zeros(len(data), dtype=torch.float32, device=padded_step_retrieve_entropy.device)
        retrieve_token_ratios = torch.zeros(len(data), dtype=torch.float32, device=padded_step_retrieve_entropy.device)

        np.set_printoptions(threshold=np.inf)
        print(f"padded_step_retrieve_token_lengths: {padded_step_retrieve_entropy[0:32].numpy()}")
        print(f"padded_step_retrieve_token_lengths: {padded_step_retrieve_token_lengths[0:32].numpy()}")
        print(f"padded_step_mask: {padded_step_mask[0:32].numpy()}")

        # print(f"padded_step_retrieve_token_lengths: {padded_step_retrieve_entropy[0]}, padded_step_retrieve_token_lengths: {padded_step_retrieve_token_lengths[0]}, padded_step_mask: {padded_step_mask[0]}")

        # ======= 计算第一个step的平均H_pre ========
        first_step_entropy = padded_step_retrieve_entropy[:, 0]
        # ===== safe uid handling =====
        if "uid" in data.non_tensor_batch:
            index = data.non_tensor_batch["uid"]
        else:
            # validation mode: no uid field
            index = np.arange(len(data))
        bsz = len(data)
        prompt2base = defaultdict(list)

        for i in range(bsz):
            pid = index[i]
            if first_step_entropy[i] >= 0:
                prompt2base[pid].append(first_step_entropy[i])
        uid2mean = {}
        for uid, arr in prompt2base.items():
            if len(arr) > 0:
                uid2mean[uid] = torch.mean(torch.stack(arr))
            else:
                uid2mean[uid] = torch.tensor(0.0)  # fallback

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            # add tokenizer to extra_info if not exists
            if extra_info is None or extra_info.get("tokenizer") is None:
                extra_info = {"tokenizer": self.tokenizer}

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            # ======= 计算retrieve reward =======
            retrieve_entropy = padded_step_retrieve_entropy[i]
            retrieve_mask = padded_step_mask[i]
            retrieve_token_length = padded_step_retrieve_token_lengths[i]
            first_step_entropy_mean = uid2mean[index[i]]
            # find retrieve steps
            retrieve_position = (retrieve_entropy == -1.0) & (retrieve_mask == 1.0)
            retrieve_idx = retrieve_position.nonzero(as_tuple=False).squeeze(-1)
            # compute retrieve drops and record h_pre.
            # If retrieved in the given rollout, compute rollout-based H_drop, H_pre, and token ratios
            if retrieve_idx.numel() > 0:
                gate_list, H_pre_list = [], []
                for each_retrieve_step in retrieve_idx:
                    if each_retrieve_step == 0:
                        # H_pre: meaned entropy from generated rollouts
                        H_pre = first_step_entropy_mean
                        # H_post: nearest entropy >=0 after retrieve_step
                        post_entropy = retrieve_entropy[each_retrieve_step + 1:]
                        post_mask = (post_entropy >= 0)
                        if not post_mask.any():
                            continue
                        H_post = post_entropy[post_mask][0]
                    else:
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
                    gate = torch.tanh((H_drop - 0.0) / 0.5)
                    gate = gate * H_pre
                    gate_list.append(gate)
                    # H_pre_list.append(H_pre)
                # Mean gate and H_pre for each rollout
                """if len(gate_list) > 0:
                    retrieve_gates[i] = torch.stack(gate_list).mean()
                if len(H_pre_list) > 0:
                    retrieve_rewards[i] = torch.stack(H_pre_list).mean()"""
                retrieve_rewards[i] = torch.stack(gate_list).mean()

                # Retrieve ratios
                # 每个 rollout 的 retrieve token 总长度
                retrieve_token = (retrieve_token_length * retrieve_mask).sum(dim=-1)
                # compute drop for each retrieve steps
                retrieve_token_ratios[i] = retrieve_token * 1.0 / max(valid_response_length, 1.0)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # id2score_retrieve = defaultdict(list)
        # index = data.non_tensor_batch["uid"]
        # with torch.no_grad():
        #     bsz = retrieve_rewards.shape[0]
        #     # -------- collect scores --------
        #     for i in range(bsz):
        #         id2score_retrieve[index[i]].append(retrieve_rewards[i])
        # print(f"id2score_retrieve: {id2score_retrieve}")
        # time.sleep(60)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "retrieve_gates": retrieve_gates,
                "retrieve_rewards": retrieve_rewards,
                "retrieve_token_ratios": retrieve_token_ratios,
            }
        else:
            return reward_tensor


class NaiveRewardManager_backup:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # ===== retrieve related tensors (batch-level) =====
        padded_step_retrieve_entropy = data.batch["padded_step_retrieve_entropy"]
        padded_step_retrieve_token_lengths = data.batch["padded_step_retrieve_token_lengths"]
        padded_step_mask = data.batch["padded_step_mask"]

        retrieve_gates = torch.zeros(len(data), dtype=torch.float32, device=padded_step_retrieve_entropy.device)
        retrieve_rewards = torch.zeros(len(data), dtype=torch.float32, device=padded_step_retrieve_entropy.device)
        retrieve_token_ratios = torch.zeros(len(data), dtype=torch.float32, device=padded_step_retrieve_entropy.device)

        np.set_printoptions(threshold=np.inf)
        print(f"padded_step_retrieve_token_lengths: {padded_step_retrieve_entropy[0:32].numpy()}")
        print(f"padded_step_retrieve_token_lengths: {padded_step_retrieve_token_lengths[0:32].numpy()}")
        print(f"padded_step_mask: {padded_step_mask[0:32].numpy()}")

        # print(f"padded_step_retrieve_token_lengths: {padded_step_retrieve_entropy[0]}, padded_step_retrieve_token_lengths: {padded_step_retrieve_token_lengths[0]}, padded_step_mask: {padded_step_mask[0]}")

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            # add tokenizer to extra_info if not exists
            if extra_info is None or extra_info.get("tokenizer") is None:
                extra_info = {"tokenizer": self.tokenizer}

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            # ======= 计算retrieve reward =======
            retrieve_entropy = padded_step_retrieve_entropy[i]
            retrieve_mask = padded_step_mask[i]
            retrieve_token_length = padded_step_retrieve_token_lengths[i]
            # find retrieve steps
            retrieve_position = (retrieve_entropy == -1.0) & (retrieve_mask == 1.0)
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
                    gate = torch.tanh((H_drop - 0.0) / 0.5)
                    gate = gate * H_pre
                    gate_list.append(gate)
                    # H_pre_list.append(H_pre)
                # Mean gate and H_pre for each rollout
                """if len(gate_list) > 0:
                    retrieve_gates[i] = torch.stack(gate_list).mean()
                if len(H_pre_list) > 0:
                    retrieve_rewards[i] = torch.stack(H_pre_list).mean()"""
                retrieve_rewards[i] = torch.stack(gate_list).mean()

                # Retrieve ratios
                # 每个 rollout 的 retrieve token 总长度
                retrieve_token = (retrieve_token_length * retrieve_mask).sum(dim=-1)
                # compute drop for each retrieve steps
                retrieve_token_ratios[i] = retrieve_token * 1.0 / max(valid_response_length, 1.0)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # id2score_retrieve = defaultdict(list)
        # index = data.non_tensor_batch["uid"]
        # with torch.no_grad():
        #     bsz = retrieve_rewards.shape[0]
        #     # -------- collect scores --------
        #     for i in range(bsz):
        #         id2score_retrieve[index[i]].append(retrieve_rewards[i])
        # print(f"id2score_retrieve: {id2score_retrieve}")
        # time.sleep(60)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "retrieve_gates": retrieve_gates,
                "retrieve_rewards": retrieve_rewards,
                "retrieve_token_ratios": retrieve_token_ratios,
            }
        else:
            return reward_tensor