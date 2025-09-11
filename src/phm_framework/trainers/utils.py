import importlib
import phmd
from phmd import datasets


def get_task(dataset, task, model_creator):
    ds = datasets.Dataset(dataset)
    task = ds[task].meta

    try:
        task_type = getattr(importlib.import_module(model_creator.__module__), 'TASK_TYPE')
        task['type'] = task_type
    except:
        pass

    return task

