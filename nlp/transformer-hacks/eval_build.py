import json
import solcx

VERSIONS = ["0.8.19", "0.8.24"]

BASES = {
    "erc20": {
        "a": """pragma solidity ^0.8.0;
contract T {
  mapping(address=>uint256) public balanceOf;
  uint256 public totalSupply;
  function transfer(address to, uint256 v) external returns (bool) {
    require(balanceOf[msg.sender] >= v, "bal");
    balanceOf[msg.sender] -= v; balanceOf[to] += v; return true;
  }
  function mint(uint256 v) external { balanceOf[msg.sender] += v; totalSupply += v; }
}""",
        "b": """pragma solidity ^0.8.0;
contract Token {
  uint256 public totalSupply;
  mapping(address=>uint256) public balanceOf;
  function mint(uint256 amount) external { balanceOf[msg.sender] += amount; totalSupply += amount; }
  function transfer(address dst, uint256 amount) external returns (bool) {
    require(balanceOf[msg.sender] >= amount, "insufficient");
    balanceOf[msg.sender] -= amount; balanceOf[dst] += amount; return true;
  }
}""",
    },
    "erc721": {
        "a": """pragma solidity ^0.8.0;
contract N {
  mapping(uint256=>address) public ownerOf;
  mapping(address=>uint256) public balanceOf;
  function mint(address to, uint256 id) external { require(ownerOf[id]==address(0)); ownerOf[id]=to; balanceOf[to]+=1; }
  function transferFrom(address from, address to, uint256 id) external {
    require(ownerOf[id]==from); ownerOf[id]=to; balanceOf[from]-=1; balanceOf[to]+=1;
  }
}""",
        "b": """pragma solidity ^0.8.0;
contract Nft {
  mapping(address=>uint256) public balanceOf;
  mapping(uint256=>address) public ownerOf;
  function transferFrom(address src, address dst, uint256 tokenId) external {
    require(ownerOf[tokenId]==src); ownerOf[tokenId]=dst; balanceOf[src]-=1; balanceOf[dst]+=1;
  }
  function mint(address dst, uint256 tokenId) external { require(ownerOf[tokenId]==address(0)); ownerOf[tokenId]=dst; balanceOf[dst]+=1; }
}""",
    },
    "proxy": {
        "a": """pragma solidity ^0.8.0;
contract P {
  address public impl;
  constructor(address i){ impl=i; }
  fallback() external payable {
    address t=impl;
    assembly {
      calldatacopy(0,0,calldatasize())
      let r:=delegatecall(gas(),t,0,calldatasize(),0,0)
      returndatacopy(0,0,returndatasize())
      switch r case 0 {revert(0,returndatasize())} default {return(0,returndatasize())}
    }
  }
}""",
        "b": """pragma solidity ^0.8.0;
contract Forwarder {
  address public target;
  constructor(address t){ target=t; }
  fallback() external payable {
    address dst=target;
    assembly {
      calldatacopy(0,0,calldatasize())
      let ok:=delegatecall(gas(),dst,0,calldatasize(),0,0)
      returndatacopy(0,0,returndatasize())
      switch ok case 0 {revert(0,returndatasize())} default {return(0,returndatasize())}
    }
  }
}""",
    },
    "storage": {
        "a": """pragma solidity ^0.8.0;
contract S { uint256 v; function set(uint256 x) external { v=x; } function get() external view returns(uint256){ return v; } }""",
        "b": """pragma solidity ^0.8.0;
contract Store { uint256 value; function get() external view returns(uint256){ return value; } function set(uint256 n) external { value=n; } }""",
    },
    "counter": {
        "a": """pragma solidity ^0.8.0;
contract C { uint256 public count; function inc() external { count++; } function dec() external { count--; } }""",
        "b": """pragma solidity ^0.8.0;
contract Ctr { uint256 public total; function dec() external { total--; } function inc() external { total++; } }""",
    },
}


def compile_runtime(src, ver):
    out = solcx.compile_source(src, output_values=["bin-runtime"], solc_version=ver)
    for k, v in out.items():
        code = v.get("bin-runtime", "")
        if code:
            return code
    return ""


def build():
    for ver in VERSIONS:
        try:
            solcx.install_solc(ver)
        except Exception as e:
            print("skip", ver, e)
    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    use = [v for v in VERSIONS if v in installed]
    if not use:
        raise SystemExit("no solc versions installed")

    items = []
    for group, variants in BASES.items():
        for vk, src in variants.items():
            base = f"{group}_{vk}"
            for ver in use:
                try:
                    code = compile_runtime(src, ver)
                except Exception as e:
                    print("fail", base, ver, e)
                    continue
                if not code:
                    continue
                items.append({
                    "id": f"{base}@{ver}",
                    "group": group,
                    "base": base,
                    "variant": ver,
                    "hex": code[2:] if code.startswith("0x") else code,
                })

    with open("eval_dataset.json", "w") as f:
        json.dump(items, f, indent=2)
    print("built eval_dataset.json:", len(items), "contracts")
    return items


if __name__ == "__main__":
    build()
