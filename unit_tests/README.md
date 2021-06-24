# Unit Tests

Spikey's tests are built with the standard python unit testing package.
Each type of module in Spikey has its own file with set of test cases in the unit_tests directory.
From base.py, the ModuleTest base case ensures all modules that inheret it are pickle-able and deep copy-able.
It also provides the tools to apply test cases to the whole set of such objects, eg TestSynapse.test_reset runs on all types of synapses given in the TYPES list.

## Run Tests

```bash
# From spikey root directory
pip install -r unit_tests/requirements.txt

bash unit_tests/run.sh  # for python command users
bash unit_tests/run3.sh  # for python3 command users
```
