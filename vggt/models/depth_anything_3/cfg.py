# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module
from omegaconf import OmegaConf

def create_object(config):
    params = OmegaConf.to_container(config, resolve=True)
    spec = params.pop('__object__')
    if spec.get('args', 'as_params') != 'as_params':
        raise ValueError('Only as_params is supported')
    return getattr(import_module(spec['path']), spec['name'])(**params)
