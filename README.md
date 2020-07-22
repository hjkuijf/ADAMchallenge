# ADAMchallenge
The Aneurysm Detection And segMenation Challenge: http://adam.isi.uu.nl/

Example docker containers for the ADAM Challenge. The example python script can be used to see how applications can be containerized and run within the challenge.

Python example
A detailed description of the Python example is provided here: http://adam.isi.uu.nl/methods/example-python/. When this container is run according to the commands below, TEAM-NAME=example:python, YOUR-COMMAND=python /adam_example/example.py, and TEST-ORIG/PRE are the input folders specified here: http://adam.isi.uu.nl/data/

Matlab example - this is provided from the WMH segmentation challenge 2017
A detailed description of the matlab example is provided here: http://adam.isi.uu.nl/methods/example-matlab/. When this container is run according to the commands below, TEAM-NAME=example:matlab, YOUR-COMMAND=wmhseg_example/example, and TEST-ORIG/PRE are the input folders specified here: http://wmh.isi.uu.nl/data/

In order to run matlab scripts in a container, the script has to be compiled with the matlab compiler. Within the container, we install the corresponding matlab runtime to execute the compiled script.

Docker commands
Containers submitted to the challenge will be run with the following commands:

CONTAINERID=`docker run -dit -v [TEST-ORIG]:/input/orig:ro -v [TEST-PRE]:/input/pre:ro -v /output adamchallenge/[TEAM-NAME]`
docker exec $CONTAINERID [YOUR-COMMAND]
docker cp $CONTAINERID:/output [RESULT-TEAM]
docker stop $CONTAINERID
docker rm -v $CONTAINERID