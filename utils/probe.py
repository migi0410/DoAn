import importlib.metadata

packages = sorted([f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()])
for pkg in packages:
    print(pkg)
