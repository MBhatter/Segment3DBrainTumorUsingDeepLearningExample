import subprocess
import os

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--imagefile",
                  action="store", dest="imagefile", default=None,
                  help="FILE containing image info", metavar="FILE")
(options, args) = parser.parse_args()


if (options.imagefile != None ):
  rootdir = '/'.join(options.imagefile.split('/')[0:-1])
  filename = options.imagefile.split('/').pop()
  contrast = filename.split('.')[0]
  outputimage = '%s.scaled.nii.gz'%(contrast)
  getHeaderCmd = 'c3d %s -dup -scale 0.0 -lstat  ' % (options.imagefile )
  print getHeaderCmd
  os.system( getHeaderCmd )
  headerProcess = subprocess.Popen(getHeaderCmd ,shell=True,stdout=subprocess.PIPE )
  while ( headerProcess.poll() == None ):
     pass
  rawlstatheader = filter(len,headerProcess.stdout.readline().strip('\n').split(" "))
  rawlstatinfo = [filter(len,lines.strip('\n').split(" ")) for lines in headerProcess.stdout.readlines()]
  labeldictionary =  dict([(int(line[0]),dict(zip(rawlstatheader[1:-1],map(float,line[1:-3])))) for line in rawlstatinfo ])
  #print labeldictionary 
  # zscore = (image - mean) / std
  rescalecmd = 'c3d -verbose %s -shift %12.5e -scale %12.5e -clip -5 5  -type float -o %s/%s ' % (options.imagefile,-labeldictionary[0]['Mean'],1./labeldictionary[0]['StdD'] , rootdir,outputimage )
  print rescalecmd 
  os.system(rescalecmd)
  verifyrescalecmd = 'c3d %s/%s -dup -scale 0.0 -lstat  ' % (rootdir,outputimage )
  print verifyrescalecmd 
  os.system( verifyrescalecmd  )
      
else:
  parser.print_help()
  print options
 
