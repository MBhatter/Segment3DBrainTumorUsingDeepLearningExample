%  @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
%   base class for image segmentation

classdef ImageSegmentationBaseClass  < handle
   properties
      % Value {mustBeNumeric}
      lgraph %  @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
             %  TODO - here lgraph is a scalar, can we make this a vector ? I want to load multiple networks 
      tabledb  %  
      jsonData %  
   end
%%    % abstact base class methods
%%    methods (Abstract)
%%       preprocess(obj,filename) % derived/inherited classes will define the preprocessing 
%%       % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR  - what other NN methods should we add ? 
%%       loadneuralnet(obj,NumberChannels) % derived/inherited classes will define the architecture
%%    end
   methods

      function obj = ImageSegmentationBaseClass(fname)
        % constructor - load all configuration data
        jsonText = fileread(fname);
        obj.jsonData = jsondecode(jsonText);
        
        % Read file pathways into table
        fullFileName = obj.jsonData.fullFileName;
        delimiter = obj.jsonData.delimiter;
        obj.tabledb = readtable(fullFileName, 'Delimiter', delimiter);

        % initialize NN
        obj.lgraph = layerGraph();
      end

   end
end


