Insallation:

first install uv in system using (refer to uv download page in google for info) in a powershell terminal run the following command to install uv:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

after installation is done, you can check if uv is installed correctly by running the following command in your terminal:
uv --version

then go to the directory where you have your new project folder created (just folder to be created) and run the following command to initialize uv:
uv init
--this will install all the necessary dependencies

in the same directory, run code. to open vs code

you will see that pyproject.toml file is present where you need to add all your dependencies and other configurations for your project.

once added, go to the cmd in that project folder and run the following command to install all the packages:
uv sync

once all packages are downloaded, activate your venv by running the following command in your terminal:
.venv\Scripts\activate

open notebook:
python -m notebook

