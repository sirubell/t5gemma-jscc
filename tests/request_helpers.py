"""Write explicit one-operation requests used by real CPU/mocked consumer tests."""
from pathlib import Path
import uuid
from typing import Any
import yaml
from jscc.experiment import main, SCHEMA


def invoke(directory, workflow, operation, execution, science=None) -> Any:
    request = {'schema': SCHEMA, 'workflow': workflow, 'operation': operation, 'execution': execution}
    if science is not None:
        request['experiment'] = science
    filename = Path(directory)/('request-'+uuid.uuid4().hex+'.yaml')
    filename.write_text(yaml.safe_dump(request, sort_keys=False))
    return main(['--config', str(filename)])
