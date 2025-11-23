from dataclasses import dataclass, asdict, fields


@dataclass
class BaseArguments:
    @property
    def yaml_exclude(self):
        return set()

    def to_dict(self):
        return asdict(self)

    def to_yaml(self):
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith("_") and f.name not in self.yaml_exclude
        }
