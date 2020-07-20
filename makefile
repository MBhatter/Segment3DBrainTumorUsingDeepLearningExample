# file : Makefile (UNIX)
#
# You can invoke this Makefile using
#  make -f Makefile MATLABROOT=[directory where MATLAB is installed]
#
# If you do not want to specify the MATLABROOT at the gmake command line
# every single time, you can define it by uncommenting the line below
# and assigning the correct root of MATLAB (with no trailing '/') on
# the right hand side.
#
MATLABROOT      := /data/apps/MATLAB/R2019a/
#

#
# Defaults
#

MEX=$(MATLABROOT)/bin/mex
MCC=$(MATLABROOT)/bin/mcc


# The following are the definitions for each target individually.

applymodel: applymodel.m
	$(MCC) -d './' -R -nodisplay -R '-logfile,./matlab.log' -S -v -m $^ $(CTF_ARCHIVE)  -o $@

tags: 
	ctags -R *


CTF_ARCHIVE=$(addprefix -a ,$(SOURCE_FILES))



SOURCE_FILES  = dicePixelClassification3dLayer.m
#$(MATLABROOT)/toolbox/nnet/


#activations.m  activationsMIMO.m  calculateActivations.m  classifyAndUpdateState.m  classify.m  DAGNetwork.m  predictAndUpdateState.m  predict.m  predictMIMO.m  predictRNN.m
