dataset_roots = {
    "VizWiz": "./data/VizWiz/",
    "MMSafety": "/home/linear_probing/data/MM-SafetyBench",
    "MAD": "/root/autodl-fs/val2017",   # MADBench uses COCO images
    "MathVista": "./data/MathVista/",
    "POPE": "/root/autodl-fs/coco_images",  # POPE uses COCO images
    "ImageNet": "./data/ImageNet/"
}


def build_dataset(dataset_name, split, prompter):
    if dataset_name == "VizWiz":
        from .VizWiz import VizWizDataset
        dataset = VizWizDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "MAD":
        from .MADBench import MADBench
        dataset = MADBench(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "ImageNet":
        from .ImageNet import ImageNetDataset
        dataset = ImageNetDataset(split, dataset_roots[dataset_name])
    elif dataset_name == "MathVista":
        from .MathV import MathVista
        dataset = MathVista(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "MMSafety":
        from .MMSafety import MMSafetyBench
        dataset = MMSafetyBench(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "POPE":
        from .POPE import POPEDataset
        dataset = POPEDataset(split, dataset_roots[dataset_name])
    elif dataset_name == 'trivia_qa':
        from .TriviaQA import TriviaQA
        dataset = TriviaQAHalo(prompter)
    elif dataset_name == 'trivia_qa_halo':
        from .TriviaQA_halo import TriviaQAHalo
        dataset = TriviaQAHalo(prompter)
    else:
        raise ValueError()

    return dataset.get_data()
