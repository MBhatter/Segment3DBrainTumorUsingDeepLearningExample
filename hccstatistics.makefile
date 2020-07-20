SHELL := /bin/bash
DATADIR = /rsrch1/ip/dtfuentes/github/hccdetection/anonymize/
WORKDIR = hccmrilog/ImageDatabase/
include hccmri512kfold005.makefile
mask:        $(addprefix $(WORKDIR),$(addsuffix /unet3d/label.nii.gz,$(UIDLIST)))
overlap:     $(addprefix $(WORKDIR)/,$(addsuffix /$(DATABASEID)/overlap.sql,$(UIDLIST)))
scaled:   $(addprefix $(WORKDIR)/,$(addsuffix /Art.scaled.nii.gz,$(UIDLIST)))  
combined: $(addprefix $(WORKDIR)/,$(addsuffix /Art.combined.nii.gz,$(UIDLIST)))  

## pre processing
$(WORKDIR)/%/Art.scaled.nii.gz: $(DATADIR)/%/Art.raw.nii.gz
	mkdir -p $(@D); python normalization.py --imagefile=$<  --output=$@
%/Art.combined.nii.gz: %/Art.scaled.nii.gz %/Truth.nii.gz
	c3d $^ -binarize  -omc $@
## dice statistics
$(WORKDIR)/%/$(DATABASEID)/overlap.csv: $(WORKDIR)/%/$(DATABASEID)/tumor.nii.gz
	mkdir -p $(@D)
	$(C3DEXE) $<  -as A $(DATADIR)/$*/TruthVen1.nii.gz -as B -overlap 1 -overlap 2 -overlap 3 -overlap 4  -thresh 2 3 1 0 -comp -as C  -clear -push C -replace 0 255 -split -pop -foreach -push B -multiply -insert A 1 -overlap 1 -overlap 2 -overlap 3 -overlap 4 -pop -endfor
	grep "^OVL" $(@D)/overlap.txt  |sed "s/OVL: \([0-9]\),/\1,$(subst /,\/,$*),/g;s/OVL: 1\([0-9]\),/1\1,$(subst /,\/,$*),/g;s/^/TruthVen1.nii.gz,$(DATABASEID)\/tumor.nii.gz,/g;"  | sed "1 i FirstImage,SecondImage,LabelID,InstanceUID,MatchingFirst,MatchingSecond,SizeOverlap,DiceSimilarity,IntersectionRatio" > $@

$(WORKDIR)/%/overlap.sql: $(WORKDIR)/%/overlap.csv
	-sqlite3 $(SQLITEDB)  -init .loadcsvsqliterc ".import $< overlap"

