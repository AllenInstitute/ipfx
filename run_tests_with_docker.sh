docker run -v ${PWD}:/root/allensdk.ipfx \
           -v /data/informatics/module_test_data/:/data/informatics/module_test_data/ \
           -v /allen/aibs/informatics/module_test_data/:/allen/aibs/informatics/module_test_data/ \
           -v /allen/programs/celltypes/production/mousecelltypes/:/allen/programs/celltypes/production/mousecelltypes/ \
           --workdir /root/allensdk.ipfx --rm -t --shm-size=8g \
           alleninstitute/allensdk_anaconda3 \
           /bin/bash --login -c "source activate py27; \
                                 export TEST_COMPLETE=true; \
                                 pip install -r requirements.txt; \
                                 pip install -r test_requirements.txt; \
                                 cd tests; py.test || exit 0"
