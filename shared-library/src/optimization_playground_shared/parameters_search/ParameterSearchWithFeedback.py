from optimization_playground_shared.parameters_search.Parameter import Parameter
from .ParameterSearch import ParameterSearch
import json
import torch
from io import BytesIO
import torchvision
import base64
import os
from copy import deepcopy

class StateFile:
    def __init__(self) -> None:
        self.state = {}

    def add_float(self, name, value):
        if torch.is_tensor(value):
            self.state[name] = value.item()
        else:
            self.state[name] = value

    def add_image(self, name, value):
        buf = BytesIO()
        torchvision.utils.save_image(value, buf, format="png")
        self.state[name] = {
            # we base64 encode images 
            "type": "image",
            "value": base64.b64encode(buf.getbuffer()).decode("ascii")
        }

class ParameterSearchState(ParameterSearch):
    def __init__(self, parameters: Parameter) -> None:
        super().__init__(parameters)

    def store_metadata_file(self):
        # We need a simple metadata file for building the sliders etc
        # and knowing where to look for files 
        metadata_file = {}
        for i in self._parameters:
            metadata_file[i.name] = i.get_values()
        
        name_config = f".parameter_search/metadata.json"
        self._make_dirs(name_config)
        with open(name_config, "w") as file:
            json.dump(metadata_file, file)

    def store_state(self, epoch, state: StateFile):
        # epoch + parameter config 
        name_value = self.parameters()
        # TODO: Figure out a better name for this ? 
        name_config = ParameterSearchState.get_file_name(epoch, name_value)
        self._make_dirs(name_config)
        with open(name_config, "w") as file:
            json.dump(state.state, file)

    @staticmethod
    def get_file_name(epoch, parameters):
        name_value = "_".join([
            f"{key}_{value}" for key, value in sorted(parameters.items(), key=lambda x: x[0])
        ])
        name_config = f".parameter_search/{epoch}/{name_value}.json"
        return name_config

    def _make_dirs(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

class ParameterSearchWithFeedback(ParameterSearchState):
    def __init__(self, parameters: Parameter) -> None:
        super().__init__(parameters)
        self.store_metadata_file()

    def all(self):
        # iterate over all the parameters -> Create new entry to ParameterSearchState
        items = []
        while not self.done:
            items.append(ParameterSearchState(deepcopy(self._parameters)))
            self.step()
        return items
