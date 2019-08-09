# TODO: add dependencies

class ConfigOpts(object):
    def __init__(self, d):
        self.reinit(d)

    def reset(self):
        for cfg in self._configs.values():
            cfg.reset()

    def clone(self):
        new_opts = ConfigOpts(self.default_dict())
        new_opts.set_values(self.value_dict())
        return new_opts

    def reinit(self, defaults):
        super().__setattr__('_configs', {})
        for k, x in defaults.items():
            if isinstance(x, tuple):
                self._configs[k] = ConfigItem(k, x[0], x[1])
            else:
                self._configs[k] = ConfigItem(k, x, type(x))

    def set_values(self, values):
        self.reset()
        for k, val in values.items():
            self._configs[k].value = val

    def set_as_default(self, values=None):
        # attributes in `values` dict must already exist
        if values is not None:
            self.set_values(values)
        for cfg in self._configs.values():
            if cfg._value is not None:
                cfg.default = cfg._value
                cfg.reset()

    def value_dict(self):
        res = {}
        for k, cfg in self._configs.items():
            if cfg._value is not None:
                res[k] = cfg.value
        return res

    def default_dict(self):
        res = {}
        for k, cfg in self._configs.items():
            res[k] = (cfg.default, cfg.type)
        return res

    def _get(self, attr):
        if attr in self._configs:
            return self._configs[attr].value
        else:
            raise KeyError(attr + ' not found in configs!')

    def _set(self, attr, val):
        if attr in self._configs:
            if val is None:
                self._configs[attr].reset()
            else:
                self._configs[attr].value = val
        else:
            raise KeyError(attr + ' not found in configs!')

    def __getattr__(self, attr):
        return self._get(attr)

    def __setattr__(self, attr, val):
        self._set(attr, val)

    def __getitem__(self, attr):
        return self._get(attr)

    def __setitem__(self, attr, val):
        self._set(attr, val)


    def __str__(self):
        res = 'ConfigOpts:\n'
        for cfg in self._configs.values():
            res += str(cfg) + '\n'
        return res

    def __iter__(self):
        return iter(self._configs.items())

    def __len__(self):
        return len(self._configs)


class ConfigItem(object):
    def __init__(self, name, default, data_type):
        assert (isinstance(data_type, type) or isinstance(data_type, list)), \
            'invalid data type!'
        self.name = name
        self.type = data_type
        self.default = default
        self._value = None

    def _typecheck(self, x):
        if isinstance(self.type, list):
            return x in self.type
        else:
            return isinstance(x, self.type)

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, x):
        if not self._typecheck(x):
            raise TypeError(str(x) + ' has wrong type! Should be ' + str(self.type))
        self._default = x

    @property
    def value(self):
        if self._value is None:
            return self._default
        else:
            return self._value

    @value.setter
    def value(self, x):
        if not self._typecheck(x):
            raise TypeError(str(x) + ' has wrong type! Should be ' + str(self.type))
        self._value = x

    def reset(self):
        self._value = None

    def __str__(self):
        return '- {}: {}'\
            .format(self.name, self.value)
