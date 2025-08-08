#!/bin/bash

set -e

docker run                                                                                                       \
    --env-file ~/env.list                                                                                        \
    --mount type=bind,source=$PWD,target=/local1/github_worker,bind-propagation=rshared                          \
    -v /data/informatics/module_test_data/:/data/informatics/module_test_data/                                   \
    -v /allen/aibs/informatics/module_test_data/:/allen/aibs/informatics/module_test_data/                       \
    -v /allen/programs/celltypes/production/mousecelltypes/:/allen/programs/celltypes/production/mousecelltypes/ \
    -v /allen/programs/celltypes/workgroups/279/:/allen/programs/celltypes/workgroups/279/                       \
    -v /allen/programs/celltypes/production/humancelltypes/:/allen/programs/celltypes/production/humancelltypes/ \
    --workdir /local1/github_worker --rm                                                                         \
    --user 1001:1001                                                                                             \
    ${DOCKER_IMAGE}                                                                                              \
      /bin/bash -c "python -m venv .venv;                                                     \
                    source .venv/bin/activate;                                                \
                    pip install --upgrade pip;                                                \
                    pip install numpy;                                                        \
                    pip install -r requirements.txt;                                          \
                    export TEST_API_ENDPOINT=http://api.brain-map.org;                        \
                    export SKIP_LIMS=false;                                                   \
                    export TEST_INHOUSE=true;                                                 \
                    export ALLOW_TEST_DOWNLOADS=true;                                         \
                    export IPFX_TEST_TIMEOUT=60;                                              \
                    pip install -r requirements-test.txt;                                     \
                    git config lfs.url 'https://github.com/AllenInstitute/ipfx.git/info/lfs'; \
                    git lfs env;                                                              \
                    git lfs pull;                                                             \
                    pip install -e .;                                                         \
                    python -m pytest --junitxml=test-reports/test.xml --verbose"
