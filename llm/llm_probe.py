import numpy as np
from llm.prompt_loader import PromptLoader
import torch
from utils.tools import create_folder, get_short_name
from utils.arguments import Arguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from baukit import Trace, TraceDict
import llama_iti
from llm.ml_tools import svd_embed_score
from sklearn.decomposition import PCA
from metric_utils import get_measures, print_measures
from linear_probe import get_linear_acc
import os


def llm_probe(dataset, used_indices):
    args = Arguments()
    cfg = args.get_config()
    num_gene = cfg.num_gene
    most_likely = cfg.most_likely
    model_name = cfg.model_name
    thres_gt = cfg.thres_gt
    wild_ratio = cfg.wild_ratio
    weighted_svd = cfg.weighted_svd

    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
    #                                                    torch_dtype=torch.float16,
    #                                                    device_map="auto").cuda()

    tokenizer = llama_iti.LlamaTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    model = llama_iti.LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
                                                       torch_dtype=torch.float16,
                                                       device_map="auto").cuda()

    short_name = get_short_name(model_name)
    use_rouge = cfg.use_rouge
    dataset_name = cfg.dataset_name

    '''#########################Extract internal states##############################'''
    # firstly get the embeddings of the generated question and answers.
    embed_generated = []  # get hidden_states for all answers

    if dataset_name == 'tydiqa':
        length = len(used_indices)
        indices = used_indices
    else:
        length = len(dataset)
        indices = [x for x in range(length)]

    prompt_loader = PromptLoader(dataset_name, dataset)

    for i in tqdm(range(length)):
        if not os.path.exists(f'./save_for_eval/{dataset_name}_hal_det/answers/most_likely_hal_det_{short_name}_{dataset_name}_answers_index_{i}.npy'):
            length = i
            break
        prompt = prompt_loader.get_prompt(int(indices[i]))
        answers = np.load(
            f'save_for_eval/{dataset_name}_hal_det/answers/most_likely_hal_det_{short_name}_{dataset_name}_answers_index_{i}.npy')

        for anw in answers:
            encoded_prompt = tokenizer(
                prompt+anw, return_tensors='pt').input_ids.cuda()

            with torch.no_grad():
                hidden_states = model(
                    encoded_prompt, output_hidden_states=True).hidden_states  # tuple(tensor,...). len(tuple)= number of layers=33. tensor.shape=(1,49(query_len),4096)
                # tensor.shape=(33 layers,49(query_len),4096)
                hidden_states = torch.stack(hidden_states, dim=0).squeeze()
                hidden_states = hidden_states.detach().cpu().numpy(
                )[:, -1, :]  # only using last token embedding
                embed_generated.append(hidden_states)

    # stack hidden_states for all answers
    embed_generated = np.asarray(np.stack(embed_generated), dtype=np.float32)
    np.save(
        f'save_for_eval/{dataset_name}_hal_det/most_likely_{short_name}_gene_embeddings_layer_wise.npy', embed_generated)

    # HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(
    #    model.config.num_hidden_layers)]
    # MLPS = [f"model.layers.{i}.mlp" for i in range(
    #    model.cfg.num_hidden_layers)]

    # embed_generated_loc2 = []  # get more internal states
    # embed_generated_loc1 = []  # get more internal states
    # for i in tqdm(range(length)):
    #    prompt = prompt_loader.get_prompt(int(indices[i]))
    #    answers = np.load(
    #        f'save_for_eval/{dataset_name}_hal_det/answers/most_likely_hal_det_{short_name}_{dataset_name}_answers_index_{i}.npy', allow_pickle=True)

    #    for anw in answers:
    #        encoded_prompt = tokenizer(
    #            prompt+anw, return_tensors='pt').input_ids.cuda()

    #        # what's the purpose? Get intermediate results from models
    #        with torch.no_grad():
    #            with TraceDict(model, HEADS + MLPS) as ret:
    #                output = model(encoded_prompt, output_hidden_states=True)
    #            head_wise_hidden_states = [
    #                ret[head].output.squeeze().detach().cpu() for head in HEADS]
    #            head_wise_hidden_states = torch.stack(
    #                head_wise_hidden_states, dim=0).squeeze().numpy()
    #            mlp_wise_hidden_states = [
    #                ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
    #            mlp_wise_hidden_states = torch.stack(
    #                mlp_wise_hidden_states, dim=0).squeeze().numpy()

    #            embed_generated_loc2.append(mlp_wise_hidden_states[:, -1, :])
    #            embed_generated_loc1.append(head_wise_hidden_states[:, -1, :])
    # embed_generated_loc2 = np.asarray(
    #    np.stack(embed_generated_loc2), dtype=np.float32)
    # embed_generated_loc1 = np.asarray(
    #    np.stack(embed_generated_loc1), dtype=np.float32)

    # np.save(
    #    f'save_for_eval/{dataset_name}_hal_det/most_likely_{short_name}_gene_embeddings_head_wise.npy', embed_generated_loc1)
    # np.save(
    #    f'save_for_eval/{dataset_name}_hal_det/most_likely_{short_name}_embeddings_mlp_wise.npy',  embed_generated_loc2)

    '''############################################################################'''

    # get the split and label (true or false) of the unlabeled data and the test data.
    if use_rouge:
        if most_likely:
            gts = np.load(f'./ml_{dataset_name}_rouge_score.npy')
        else:
            gts_bg = np.load(f'./bg_{dataset_name}_rouge_score.npy')
    else:
        if most_likely:
            gts = np.load(f'./ml_{dataset_name}_bleurt_score.npy')
        else:
            gts_bg = np.load(f'./bg_{dataset_name}_bleurt_score.npy')
    thres = thres_gt
    # get ground truth labels
    gt_label = np.asarray(gts > thres, dtype=np.int32)

    permuted_index = np.random.permutation(length)
    wild_q_indices = permuted_index[:int(
        wild_ratio * length)]  # train and valid set

    validset_len = 300
    # exclude validation samples.
    wild_q_indices1 = wild_q_indices[:len(
        wild_q_indices) - validset_len]  # training set
    wild_q_indices2 = wild_q_indices[len(
        wild_q_indices) - validset_len:]  # test set
    gt_label_test = []
    gt_label_wild = []
    gt_label_val = []

    for i in range(length):
        if i not in wild_q_indices:
            gt_label_test.extend(gt_label[i: i+1])
        elif i in wild_q_indices1:
            gt_label_wild.extend(gt_label[i: i+1])
        else:
            gt_label_val.extend(gt_label[i: i+1])

    '''get testset, wildset and valset. The valset is used for determining the hype-parameters'''
    gt_label_test = np.asarray(gt_label_test)
    gt_label_wild = np.asarray(gt_label_wild)
    gt_label_val = np.asarray(gt_label_val)

    print(f'trainset length:{len(gt_label_wild)}')
    print(f'validset length:{len(gt_label_val)}')
    print(f'testset length:{len(gt_label_test)}')
    print(
        f'testset postive: {sum(gt_label_test==1)} negative: {sum(gt_label_test==0)}')

    ''''''
    feat_loc = cfg.feat_loc_svd
    if most_likely:
        if feat_loc == 3:
            embed_generated = np.load(f'save_for_eval/{dataset_name}_hal_det/most_likely_{short_name}_gene_embeddings_layer_wise.npy',
                                      allow_pickle=True)
        elif feat_loc == 2:
            embed_generated = np.load(
                f'save_for_eval/{dataset_name}_hal_det/most_likely_{short_name}_gene_embeddings_mlp_wise.npy',
                allow_pickle=True)
        else:
            embed_generated = np.load(
                f'save_for_eval/{dataset_name}_hal_det/most_likely_{short_name}_gene_embeddings_head_wise.npy',
                allow_pickle=True)
        feat_indices_wild = []
        feat_indices_eval = []

        for i in range(length):
            if i in wild_q_indices1:
                feat_indices_wild.extend(np.arange(i, i+1).tolist())
            elif i in wild_q_indices2:
                feat_indices_eval.extend(np.arange(i, i + 1).tolist())
        if feat_loc == 3:
            embed_generated_wild = embed_generated[feat_indices_wild][:, 1:, :]
            embed_generated_eval = embed_generated[feat_indices_eval][:, 1:, :]
        else:
            embed_generated_wild = embed_generated[feat_indices_wild]
            embed_generated_eval = embed_generated[feat_indices_eval]

    # k_span is the max k value

    # graid search for best hyper-parameters on validation set, and store them into returned_results
    returned_results = svd_embed_score(embed_generated_eval, gt_label_val,
                                       begin_k=1, k_span=11, mean=0, svd=0, weight=weighted_svd)

    pca_model = PCA(n_components=returned_results['k'], whiten=False).fit(
        embed_generated_wild[:, returned_results['best_layer'], :])
    projection = pca_model.components_.T
    if weighted_svd:
        projection = pca_model.singular_values_ * projection
    scores = np.mean(np.matmul(
        embed_generated_wild[:, returned_results['best_layer'], :], projection), -1, keepdims=True)
    assert scores.shape[1] == 1
    best_scores = np.sqrt(np.sum(np.square(scores), axis=1)
                          ) * returned_results['best_sign']

    '''get score for direct projection'''
    # direct projection
    feat_indices_test = []
    for i in range(length):
        if i not in wild_q_indices:
            feat_indices_test.extend(np.arange(1 * i, 1 * i + 1).tolist())
    if feat_loc == 3:
        embed_generated_test = embed_generated[feat_indices_test][:, 1:, :]
    else:
        embed_generated_test = embed_generated[feat_indices_test]

    test_scores = np.mean(np.matmul(embed_generated_test[:, returned_results['best_layer'], :],
                                    projection), -1, keepdims=True)

    assert test_scores.shape[1] == 1
    test_scores = np.sqrt(np.sum(np.square(test_scores), axis=1))

    measures = get_measures(returned_results['best_sign'] * test_scores[gt_label_test == 1],
                            returned_results['best_sign'] * test_scores[gt_label_test == 0], plot=False)
    print_measures(measures[0], measures[1], measures[2], 'direct-projection')

    '''train linear classifier???'''
    thresholds = np.linspace(0, 1, num=40)[1:-1]

    def normalizer(x): return x / \
        (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

    # graid search
    auroc_over_thres = []
    for thres_wild in thresholds:
        best_auroc = 0
        for layer in range(len(embed_generated_wild[0])):
            thres_wild_score = np.sort(best_scores)[
                int(len(best_scores) * thres_wild)]
            true_wild = embed_generated_wild[:, layer,
                                             # get ground truth???
                                             :][best_scores > thres_wild_score]
            false_wild = embed_generated_wild[:, layer,
                                              :][best_scores <= thres_wild_score]

            embed_train = np.concatenate([true_wild, false_wild], 0)
            label_train = np.concatenate([np.ones(len(true_wild)),
                                          np.zeros(len(false_wild))], 0)

            # gt training, saplma
            # embed_train = embed_generated_wild[:,layer,:]
            # label_train = gt_label_wild
            # gt training, saplma

            best_acc, final_acc, (
                clf, best_state, best_preds, preds, labels_val), losses_train = get_linear_acc(
                embed_train,
                label_train,
                embed_train,
                label_train,
                2, epochs=50,
                print_ret=True,
                batch_size=512,
                cosine=True,
                nonlinear=True,
                learning_rate=0.05,
                weight_decay=0.0003)

            clf.eval()
            output = clf(torch.from_numpy(
                embed_generated_test[:, layer, :]).cuda())
            pca_wild_score_binary_cls = torch.sigmoid(output)

            pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()

            if np.isnan(pca_wild_score_binary_cls).sum() > 0:
                breakpoint()
            measures = get_measures(pca_wild_score_binary_cls[gt_label_test == 1],
                                    pca_wild_score_binary_cls[gt_label_test == 0], plot=False)

            if measures[0] > best_auroc:
                best_auroc = measures[0]
                best_result = [100 * measures[0]]
                best_layer = layer

        auroc_over_thres.append(best_auroc)
        print('thres: ', thres_wild, 'best result: ',
              best_result, 'best_layer: ', best_layer)
