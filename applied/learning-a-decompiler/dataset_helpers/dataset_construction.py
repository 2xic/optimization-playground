"""
Our job is to learn the following

[assembly] -> [solc]
"""

import os
import json
import subprocess
import glob
import hashlib
from opcodes_table import decode
import hashlib

# location of the solc binaries
solc_assets = "~/.solc-select/artifacts"

class SourceParser:
    def __init__(self, path):
        self.dir_name = os.path.dirname(path)
        metadata_file = os.path.join(self.dir_name, "metadata.json")
        self.main_sol = os.path.join(self.dir_name, "main.sol")

        with open(metadata_file, "r") as file:
            self.metadata = json.load(file)
        self.compiler = self.metadata["CompilerVersion"].split("+")[0].replace("v", "")
        self.compiler = f"{solc_assets}/solc-{self.compiler}/solc-{self.compiler}"

    def create_dataset_entry(self):
        for output in self._get_opcodes():
            code_hash = hashlib.sha256(output["bin"].encode()).hexdigest()
            output_file = os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "dataset",
                code_hash + ".json"
            )
            with open(output_file, "w") as file:
                json.dump(output, file)

    def _get_opcodes(self):
        command = f"{self.compiler} --combined-json bin,generated-sources {self.main_sol}"
        output = subprocess.getoutput(command)
        try:
            return self._parse_2_json(output)
        except Exception as e:
            print(e)
            return []

    def _parse_2_json(self, content):
        contract_data = []
        content = json.loads(content)
        for files in content["contracts"]:
            base = (content["contracts"][files])
            if len(base["generated-sources"]) and len(base["generated-sources"]) == 1:
                contract_data.append({
                    "bin": decode(base["bin"]),
                    "source": base["generated-sources"][0]["contents"]
                })
                print(files)
        return contract_data


files = glob.glob("smart-contract-fiesta/organized_contracts/00/**/main.sol", recursive=True)
for index, i in enumerate(files[:10_000]):
    SourceParser(i).create_dataset_entry()
    if index % 100 == 0:
        print(str(i))
