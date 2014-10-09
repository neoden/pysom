usage: somclt.py [-h] [-i INIT] -s STATE [-d DATA] [--alpha ALPHA]
                 [--radius RADIUS] [--nh NH] [--maxiter MAXITER] [-v]
                 [command]

SOM network command-line tool

positional arguments:
  command               Command to perform: init|train|clusot

optional arguments:
  -h, --help            show this help message and exit
  -i INIT, --init INIT  Init with dimensions: width*height*inputs
  -s STATE, --state STATE
                        Map state file name
  -d DATA, --data DATA  Training dataset
  --alpha ALPHA         Training function parameters: variant,arg1,arg2...
  --radius RADIUS       Radius function parameters: variant,arg1,arg2...
  --nh NH               Neighbourhood function variant
  --maxiter MAXITER     Maximum iterations
  -v, --verbose         Additional information while training
