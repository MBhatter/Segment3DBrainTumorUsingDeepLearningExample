classdef ImageSegmentationBaseClass
   properties
      Value {mustBeNumeric}
   end
   methods
      function preprocess(obj,filename)
         disp('overload me with your data specific preprocessing')
         disp(['load data from a  your csv file - ', filename])
      end
   end
end

