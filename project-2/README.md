# NLU Project 2: StoryCloze Task
Authors:
* Anders Munch [amunch@student.ethz.ch]
* Thomas Nilsson [tnilsson@student.ethz.ch]
* Joel Oskarsson [ojoel@student.ethz.ch]
* Minh Duc Tran [tranmi@student.ethz.ch]


## Install python3 requirements and download data
Run the following command:
```bash
sh setup.sh
```

If the script fails:
* If ```permission denied``` occurs, you can make it exceutable with ```chmod +x download.sh```
* The files can be downloaded manually from PolyBox at https://polybox.ethz.ch/index.php/s/UqbUKdkIQBhWzvI
* Python dependencies can be installed manually with `pip3 install -r requirements.txt`

## Run everything (requires python3)
To train the models and evaulate them, go to the `scripts` directory and run the following from your shell:
```bash
sh train_all.sh
```