
clean:
	make -C docs clean
	rm -f Pipfile.lock

full_test:
	TEST_INHOUSE=true IPFX_TEST_TIMEOUT=10.0 pytest tests/ --cov=ipfx --cov-report=html

test:
	pytest tests/ --cov=ipfx --cov-report=html
