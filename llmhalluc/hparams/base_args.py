from dataclasses import asdict, dataclass, fields


@dataclass
class BaseArguments:
    @property
    def yaml_exclude(self):
        return set()

    def to_dict(self):
        return asdict(self)

    def to_yaml(
        self,
        exclude: bool = True,
        exclude_keys: set[str] = set(),
        include_keys: set[str] = set(),
        normalized: bool = False,
    ):
        exclude_keys = (
            self.yaml_exclude | exclude_keys - include_keys if exclude else set()
        )
        return {
            f.name: normalized(getattr(self, f.name))
            if normalized
            else getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) and (f.name not in exclude_keys)
        }
