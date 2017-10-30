/**
 * @Author: Skeleton provided by Yuting Liu and Jude Shavlik.  
   @Author: Function implemented by Zhanpeng Zeng, Qinyuan Sun, Jingyi Zhao
 * 
 * Copyright 2017.  Free for educational and basic-research use.
 * 
 * The main class for Lab3 of cs638/838.
 * 
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 * 
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in Lab3.java, insert that class here to simplify grading.
 * 
 * Some snippets from Jude's code left in here - feel free to use or discard.
 *
 */

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import javax.imageio.ImageIO;

public class Lab3 {
    
	private static int     imageSize = 32; // Images are imageSize x imageSize.  The provided data is 128x128, but this can be resized by setting this value (or passing in an argument).  
	                                       // You might want to resize to 8x8, 16x16, 32x32, or 64x64; this can reduce your network size and speed up debugging runs.
	                                       // ALL IMAGES IN A TRAINING RUN SHOULD BE THE *SAME* SIZE.
	private static enum    Category { airplanes, butterfly, flower, grand_piano, starfish, watch };  // We'll hardwire these in, but more robust code would not do so.
	
	private static final Boolean    useRGB = true; // If true, FOUR units are used per pixel: red, green, blue, and grey.  If false, only ONE (the grey-scale value).
	private static       int unitsPerPixel = (useRGB ? 4 : 1); // If using RGB, use red+blue+green+grey.  Otherwise just use the grey value.
			
	private static String    modelToUse = "deep"; // Should be one of { "perceptrons", "oneLayer", "deep" };  You might want to use this if you are trying approaches other than a Deep ANN.
	private static int       inputVectorSize;         // The provided code uses a 1D vector of input features.  You might want to create a 2D version for your Deep ANN code.  
	                                                  // Or use the get2DfeatureValue() 'accessor function' that maps 2D coordinates into the 1D vector.  
	                                                  // The last element in this vector holds the 'teacher-provided' label of the example.

	private static double eta       =    0.002, fractionOfTrainingToUse = 1.00, dropoutRate = 0; // To turn off drop out, set dropoutRate to 0.0 (or a neg number).
	private static int    maxEpochs = 20; // Feel free to set to a different value.

	
	private static NeuralNet model; 
	private static PerformanceAnalyser analyser; 
	
	public static void main(String[] args) {
		String trainDirectory = "images/trainset/";
		String  tuneDirectory = "images/tuneset/";
		String  testDirectory = "images/testset/";
		
        if(args.length > 5) {
            System.err.println("Usage error: java Lab3 <train_set_folder_path> <tune_set_folder_path> <test_set_foler_path> <imageSize>");
            System.exit(1);
        }
        if (args.length >= 1) { trainDirectory = args[0]; }
        if (args.length >= 2) {  tuneDirectory = args[1]; }
        if (args.length >= 3) {  testDirectory = args[2]; }
        if (args.length >= 4) {  imageSize     = Integer.parseInt(args[3]); }
    
		// Here are statements with the absolute path to open images folder
        File trainsetDir = new File(trainDirectory);
        File tunesetDir  = new File( tuneDirectory);
        File testsetDir  = new File( testDirectory);
        
        // create three datasets
		Dataset trainset = new Dataset();
        Dataset  tuneset = new Dataset();
        Dataset  testset = new Dataset();
        
        // Load in images into datasets.
        long start = System.currentTimeMillis();
        loadDataset(trainset, trainsetDir);
        System.out.println("The trainset contains " + comma(trainset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        loadDataset(tuneset, tunesetDir);
        System.out.println("The  testset contains " + comma( tuneset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        loadDataset(testset, testsetDir);
        System.out.println("The  tuneset contains " + comma( testset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        
        // Now train a Deep ANN.  You might wish to first use your Lab 2 code here and see how one layer of HUs does.  Maybe even try your perceptron code.
        // We are providing code that converts images to feature vectors.  Feel free to discard or modify.
        start = System.currentTimeMillis();
        trainANN(trainset, tuneset, testset);
        System.out.println("\nTook " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to train.");
        
    }

	public static void loadDataset(Dataset dataset, File dir) {
        for(File file : dir.listFiles()) {
            // check all files
             if(!file.isFile() || !file.getName().endsWith(".jpg")) {
                continue;
            }
            //String path = file.getAbsolutePath();
            BufferedImage img = null, scaledBI = null;
            try {
                // load in all images
                img = ImageIO.read(file);
                // every image's name is in such format:
                // label_image_XXXX(4 digits) though this code could handle more than 4 digits.
                String name = file.getName();
                int locationOfUnderscoreImage = name.indexOf("_image");
                
                // Resize the image if requested.  Any resizing allowed, but should really be one of 8x8, 16x16, 32x32, or 64x64 (original data is 128x128).
                if (imageSize != 128) {
                    scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g = scaledBI.createGraphics();
                    g.drawImage(img, 0, 0, imageSize, imageSize, null);
                    g.dispose();
                }
                
                Instance instance = new Instance(scaledBI == null ? img : scaledBI, name, name.substring(0, locationOfUnderscoreImage));

                dataset.add(instance);
            } catch (IOException e) {
                System.err.println("Error: cannot load in the image file");
                System.exit(1);
            }
        }
    }
	///////////////////////////////////////////////////////////////////////////////////////////////
	
	private static Category convertCategoryStringToEnum(String name) {
		if ("airplanes".equals(name))   return Category.airplanes; // Should have been the singular 'airplane' but we'll live with this minor error.
		if ("butterfly".equals(name))   return Category.butterfly;
		if ("flower".equals(name))      return Category.flower;
		if ("grand_piano".equals(name)) return Category.grand_piano;
		if ("starfish".equals(name))    return Category.starfish;
		if ("watch".equals(name))       return Category.watch;
		throw new Error("Unknown category: " + name);		
	}

	private static double getRandomWeight(int fanin, int fanout) { // This is one 'rule of thumb' for initializing weights.  Fine for perceptrons and one-layer ANN at least.
		double range = Math.max(Double.MIN_VALUE, 4.0 / Math.sqrt(6.0 * (fanin + fanout)));
		return (2.0 * random() - 1.0) * range;
	}
	
	// Map from 2D coordinates (in pixels) to the 1D fixed-length feature vector.
	private static double get2DfeatureValue(Vector<Double> ex, int x, int y, int offset) { // If only using GREY, then offset = 0;  Else offset = 0 for RED, 1 for GREEN, 2 for BLUE, and 3 for GREY.
		return ex.get(unitsPerPixel * (y * imageSize + x) + offset); // Jude: I have not used this, so might need debugging.
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////////

    
	// Return the count of TESTSET errors for the chosen model.
    private static int trainANN(Dataset trainset, Dataset tuneset, Dataset testset) {
    	Instance sampleImage = trainset.getImages().get(0); // Assume there is at least one train image!
    	inputVectorSize = sampleImage.getWidth() * sampleImage.getHeight() * unitsPerPixel + 1; // The '-1' for the bias is not explicitly added to all examples (instead code should implicitly handle it).  The final 1 is for the CATEGORY.
    	
    	// For RGB, we use FOUR input units per pixel: red, green, blue, plus grey.  Otherwise we only use GREY scale.
    	// Pixel values are integers in [0,255], which we convert to a double in [0.0, 1.0].
    	// The last item in a feature vector is the CATEGORY, encoded as a double in 0 to the size on the Category enum.
    	// We do not explicitly store the '-1' that is used for the bias.  Instead code (to be written) will need to implicitly handle that extra feature.
    	System.out.println("\nThe input vector size is " + comma(inputVectorSize - 1) + ".\n");
    	
    	Vector<Vector<Double>> trainFeatureVectors = new Vector<Vector<Double>>(trainset.getSize());
    	Vector<Vector<Double>>  tuneFeatureVectors = new Vector<Vector<Double>>( tuneset.getSize());
    	Vector<Vector<Double>>  testFeatureVectors = new Vector<Vector<Double>>( testset.getSize());
		
        long start = System.currentTimeMillis();
		fillFeatureVectors(trainFeatureVectors, trainset);
        System.out.println("Converted " + trainFeatureVectors.size() + " TRAIN examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        fillFeatureVectors( tuneFeatureVectors,  tuneset);
        System.out.println("Converted " +  tuneFeatureVectors.size() + " TUNE  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
		fillFeatureVectors( testFeatureVectors,  testset);
        System.out.println("Converted " +  testFeatureVectors.size() + " TEST  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        System.out.println("\nTime to start learning!");
        
        // Call your Deep ANN here.  We recommend you create a separate class file for that during testing and debugging, but before submitting your code cut-and-paste that code here.
		
        if      ("perceptrons".equals(modelToUse)) return trainPerceptrons(trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Either comment out this line or just right a 'dummy' function.
        else if ("oneLayer".equals(   modelToUse)) return trainOneHU(      trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Ditto.
        else if ("deep".equals(       modelToUse)) return trainDeep(             trainset, tuneset, testset);
        return -1;
	}
    
	private static void fillFeatureVectors(Vector<Vector<Double>> featureVectors, Dataset dataset) {
		for (Instance image : dataset.getImages()) {
			featureVectors.addElement(convertToFeatureVector(image));
		}
	}

	private static Vector<Double> convertToFeatureVector(Instance image) {
		Vector<Double> result = new Vector<Double>(inputVectorSize);		

		for (int index = 0; index < inputVectorSize - 1; index++) { // Need to subtract 1 since the last item is the CATEGORY.
			if (useRGB) {
				int xValue = (index / unitsPerPixel) % image.getWidth(); 
				int yValue = (index / unitsPerPixel) / image.getWidth();
			//	System.out.println("  xValue = " + xValue + " and yValue = " + yValue + " for index = " + index);
				if      (index % 3 == 0) result.add(image.getRedChannel()  [xValue][yValue] / 255.0); // If unitsPerPixel > 4, this if-then-elseif needs to be edited!
				else if (index % 3 == 1) result.add(image.getGreenChannel()[xValue][yValue] / 255.0);
				else if (index % 3 == 2) result.add(image.getBlueChannel() [xValue][yValue] / 255.0);
				else                     result.add(image.getGrayImage()   [xValue][yValue] / 255.0); // Seems reasonable to also provide the GREY value.
			} else {
				int xValue = index % image.getWidth();
				int yValue = index / image.getWidth();
				result.add(                         image.getGrayImage()   [xValue][yValue] / 255.0);
			}
		}
		result.add((double) convertCategoryStringToEnum(image.getLabel()).ordinal()); // The last item is the CATEGORY, representing as an integer starting at 0 (and that int is then coerced to double).
		
		return result;
	}
	
	////////////////////  Some utility methods (cut-and-pasted from JWS' Utils.java file). ///////////////////////////////////////////////////
	
	private static final long millisecInMinute = 60000;
	private static final long millisecInHour   = 60 * millisecInMinute;
	private static final long millisecInDay    = 24 * millisecInHour;
	public static String convertMillisecondsToTimeSpan(long millisec) {
		return convertMillisecondsToTimeSpan(millisec, 0);
	}
	public static String convertMillisecondsToTimeSpan(long millisec, int digits) {
		if (millisec ==    0) { return "0 seconds"; } // Handle these cases this way rather than saying "0 milliseconds."
		if (millisec <  1000) { return comma(millisec) + " milliseconds"; } // Or just comment out these two lines?
		if (millisec > millisecInDay)    { return comma(millisec / millisecInDay)    + " days and "    + convertMillisecondsToTimeSpan(millisec % millisecInDay,    digits); }
		if (millisec > millisecInHour)   { return comma(millisec / millisecInHour)   + " hours and "   + convertMillisecondsToTimeSpan(millisec % millisecInHour,   digits); }
		if (millisec > millisecInMinute) { return comma(millisec / millisecInMinute) + " minutes and " + convertMillisecondsToTimeSpan(millisec % millisecInMinute, digits); }
		
		return truncate(millisec / 1000.0, digits) + " seconds"; 
	}

    public static String comma(int value) { // Always use separators (e.g., "100,000").
    	return String.format("%,d", value);    	
    }    
    public static String comma(long value) { // Always use separators (e.g., "100,000").
    	return String.format("%,d", value);    	
    }   
    public static String comma(double value) { // Always use separators (e.g., "100,000").
    	return String.format("%,f", value);    	
    }
    public static String padLeft(String value, int width) {
    	String spec = "%" + width + "s";
    	return String.format(spec, value);    	
    }
    
    /**
     * Format the given floating point number by truncating it to the specified
     * number of decimal places.
     * 
     * @param d
     *            A number.
     * @param decimals
     *            How many decimal places the number should have when displayed.
     * @return A string containing the given number formatted to the specified
     *         number of decimal places.
     */
    public static String truncate(double d, int decimals) {
    	double abs = Math.abs(d);
    	if (abs > 1e13)             { 
    		return String.format("%."  + (decimals + 4) + "g", d);
    	} else if (abs > 0 && abs < Math.pow(10, -decimals))  { 
    		return String.format("%."  +  decimals      + "g", d);
    	}
        return     String.format("%,." +  decimals      + "f", d);
    }
    
    /** Randomly permute vector in place.
     *
     * @param <T>  Type of vector to permute.
     * @param vector Vector to permute in place. 
     */
    public static <T> void permute(Vector<T> vector) {
    	if (vector != null) { // NOTE from JWS (2/2/12): not sure this is an unbiased permute; I prefer (1) assigning random number to each element, (2) sorting, (3) removing random numbers.
    		// But also see "http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle" which justifies this.
    		/*	To shuffle an array a of n elements (indices 0..n-1):
 									for i from n - 1 downto 1 do
      								j <- random integer with 0 <= j <= i
      								exchange a[j] and a[i]
    		 */

    		for (int i = vector.size() - 1; i >= 1; i--) {  // Note from JWS (2/2/12): to match the above I reversed the FOR loop that Trevor wrote, though I don't think it matters.
    			int j = random0toNminus1(i + 1);
    			if (j != i) {
    				T swap =    vector.get(i);
    				vector.set(i, vector.get(j));
    				vector.set(j, swap);
    			}
    		}
    	}
    }
    
    public static Random randomInstance = new Random(638 * 838);  // Change the 638 * 838 to get a different sequence of random numbers.
    
    /**
     * @return The next random double.
     */
    public static double random() {
        return randomInstance.nextDouble();
    }

    /**
     * @param lower
     *            The lower end of the interval.
     * @param upper
     *            The upper end of the interval. It is not possible for the
     *            returned random number to equal this number.
     * @return Returns a random integer in the given interval [lower, upper).
     */
    public static int randomInInterval(int lower, int upper) {
    	return lower + (int) Math.floor(random() * (upper - lower));
    }


    /**
     * @param upper
     *            The upper bound on the interval.
     * @return A random number in the interval [0, upper).
     * @see Utils#randomInInterval(int, int)
     */
    public static int random0toNminus1(int upper) {
    	return randomInInterval(0, upper);
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////  Write your own code below here.  Feel free to use or discard what is provided.
    	
	private static int trainPerceptrons(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
		Vector<Vector<Double>> perceptrons = new Vector<Vector<Double>>(Category.values().length);  // One perceptron per category.

		for (int i = 0; i < Category.values().length; i++) {
			Vector<Double> perceptron = new Vector<Double>(inputVectorSize);  // Note: inputVectorSize includes the OUTPUT CATEGORY as the LAST element.  That element in the perceptron will be the BIAS.
			perceptrons.add(perceptron);
			for (int indexWgt = 0; indexWgt < inputVectorSize; indexWgt++) perceptron.add(getRandomWeight(inputVectorSize, 1)); // Initialize weights.
		}

		if (fractionOfTrainingToUse < 1.0) {  // Randomize list, then get the first N of them.
			int numberToKeep = (int) (fractionOfTrainingToUse * trainFeatureVectors.size());
			Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(numberToKeep);

			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.
			for (int i = 0; i <numberToKeep; i++) {
				trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
			}
			trainFeatureVectors = trainFeatureVectors_temp;
		}
		
        int trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE, best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;
        long  overallStart = System.currentTimeMillis(), start = overallStart;
		
        
        Data train = new Data(); 
        train.loadData(trainFeatureVectors, 6); 
        Data tune = new Data(); 
        tune.loadData(tuneFeatureVectors, 6); 
        
        model = new NeuralNet(train, tune); 
        
        model.stuckLinearActivationLayer(dropoutRate);
		
		model.stuckFullyConnectedLayer(6);
		model.stuckSigmoidActivationLayer(0);
		
		model.finalizeConfiguration();
		
        
		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.

            // CODE NEEDED HERE!
			model.train(eta, true);
			
			System.out.print("\nTraining Loss: " + String.format("%.5f", model.trainLoss)); 
			System.out.print("\tTrain Error: " + String.format("%.5f", model.trainError)); 
			System.out.println("\tTune Error: " + String.format("%.5f", model.tuneError)); 
			
	        System.out.println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
	        reportPerceptronConfig(); // Print out some info after epoch, so you can see what experiment is running in a given console.
	        start = System.currentTimeMillis();
		}
    	System.out.println("\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size()) + " (" + truncate((100.0 *      best_tuneSetErrors) / tuneFeatureVectors.size(), 2) + "%) at epoch = " + comma(best_epoch) 
    						+ " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(testFeatureVectors.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
    	return testSetErrorsAtBestTune;
	}
	
	private static void reportPerceptronConfig() {
		System.out.println(  "***** PERCEPTRON: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) + ", eta = " + truncate(eta, 2) + ", dropout rate = " + truncate(dropoutRate, 2)	);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////   ONE HIDDEN LAYER

	private static boolean debugOneLayer               = false;  // If set true, more things checked and/or printed (which does slow down the code).
	private static int    numberOfHiddenUnits          = 250;
	
	private static int trainOneHU(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
	    long overallStart   = System.currentTimeMillis(), start = overallStart;
        int  trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE, best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;
        
        Data train = new Data(); 
        train.loadData(trainFeatureVectors, 6); 
        Data tune = new Data(); 
        tune.loadData(tuneFeatureVectors, 6); 
        Data test = new Data(); 
        test.loadData(testFeatureVectors, 6); 
        
        model = new NeuralNet(train, tune); 
        analyser = new PerformanceAnalyser(); 
        
        //model.stuckLinearActivationLayer(dropoutRate);
        
        model.stuckFullyConnectedLayer(numberOfHiddenUnits);
		//model.stuckReLUActivationLayer(dropoutRate);
		
		model.stuckFullyConnectedLayer(6);
		//model.stuckReLUActivationLayer(0);
		
		model.finalizeConfiguration();
        
		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.

            // CODE NEEDED HERE!
			model.train(eta, true);
			analyser.measure(model, test, true, epoch); 
			
	        System.out.println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
	        reportOneLayerConfig(); // Print out some info after epoch, so you can see what experiment is running in a given console.
	        start = System.currentTimeMillis();
		}
		
		System.out.println("\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size()) + " (" + truncate((100.0 *      best_tuneSetErrors) / tuneFeatureVectors.size(), 2) + "%) at epoch = " + comma(best_epoch) 
		                    + " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(testFeatureVectors.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
    	return testSetErrorsAtBestTune;
	}
	
	private static void reportOneLayerConfig() {
		
		System.out.println(  "***** ONE-LAYER: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) 
		        + ", eta = " + truncate(eta, 2)   + ", dropout rate = "      + truncate(dropoutRate, 2) + ", number HUs = " + numberOfHiddenUnits
			//	+ ", activationFunctionForHUs = " + activationFunctionForHUs + ", activationFunctionForOutputs = " + activationFunctionForOutputs
			//	+ ", # forward props = " + comma(forwardPropCounter)
				);
		
	//	for (Category cat : Category.values()) {  // Report the output unit biases.
	//		int catIndex = cat.ordinal();
    //
	//		System.out.print("  bias(" + cat + ") = " + truncate(weightsToOutputUnits[numberOfHiddenUnits][catIndex], 6));
	//	}   System.out.println();
	}

	// private static long forwardPropCounter = 0;  // Count the number of forward propagations performed.
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////  DEEP ANN Code


	private static int trainDeep(Dataset trainset, Dataset tuneset, Dataset testset) {
		// You need to implement this method!
		long overallStart   = System.currentTimeMillis(), start = overallStart;

		int totalInst = 9000; 
		
		Dataset moreTrain = moreInstances(trainset, totalInst); 
		
		Data train = new Data(); 
        Data tune = new Data(); 
        Data test = new Data(); 
        
        load(train, moreTrain); 
        load(tune, tuneset); 
        load(test, testset); 
        
        System.out.println("Training set size: " + train.size()); 
        NeuralNet model = new NeuralNet(train, tune); 
        analyser = new PerformanceAnalyser(); 
        
        model.stuckConvolutionLayer(5, 20);
        model.stuckMaxPoolingLayer();
        
        model.stuckConvolutionLayer(5, 20);
        model.stuckMaxPoolingLayer();
        
        model.stuckVectorizationLayer();

        model.stuckFullyConnectedLayer(200);
        model.stuckReLUActivationLayer(0);
        
        model.stuckFullyConnectedLayer(6);
        model.stuckReLUActivationLayer(0);
        
        model.finalizeConfiguration();
        
        model.setBatchSize(1);
        
        for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).

            // CODE NEEDED HERE!
			model.train(eta, true);
			System.out.println("Test set confusion matrix"); 
			
			dispConfusionMatrix(analyser.measure(model, test, true, epoch)); 
			
	        System.out.println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
	        start = System.currentTimeMillis();
		}
        
        System.out.println("Early stopping point test set confusion matrix"); 
        dispConfusionMatrix(PerformanceAnalyser.confusionMatrix(model.getEarlyStopModel(), test)); 
        
        return 0; 
	}
	
	private static void dispConfusionMatrix(Mat cm) {
		System.out.println("\t\t\tCORRECT Category"); 
		System.out.println("           airplanes betterfly flower piano  starfish  watch"); 
		
		System.out.print("  airplanes\t"); 
		System.out.print((int)cm.matrix[0][0] + "\t" + (int)cm.matrix[0][1] + "\t" + (int)cm.matrix[0][2] + "\t"); 
		System.out.print((int)cm.matrix[0][3] + "\t" + (int)cm.matrix[0][4] + "\t" + (int)cm.matrix[0][5] + "\t"); 
		System.out.println("row SUM = " + sum(cm.matrix[0])); 
		
		System.out.print("  betterfly\t"); 
		System.out.print((int)cm.matrix[1][0] + "\t" + (int)cm.matrix[1][1] + "\t" + (int)cm.matrix[1][2] + "\t"); 
		System.out.print((int)cm.matrix[1][3] + "\t" + (int)cm.matrix[1][4] + "\t" + (int)cm.matrix[1][5] + "\t"); 
		System.out.println("row SUM = " + sum(cm.matrix[1])); 
		
		System.out.print("     flower\t"); 
		System.out.print((int)cm.matrix[2][0] + "\t" + (int)cm.matrix[2][1] + "\t" + (int)cm.matrix[2][2] + "\t"); 
		System.out.print((int)cm.matrix[2][3] + "\t" + (int)cm.matrix[2][4] + "\t" + (int)cm.matrix[2][5] + "\t"); 
		System.out.println("row SUM = " + sum(cm.matrix[2])); 
		
		System.out.print("      piano\t"); 
		System.out.print((int)cm.matrix[3][0] + "\t" + (int)cm.matrix[3][1] + "\t" + (int)cm.matrix[3][2] + "\t"); 
		System.out.print((int)cm.matrix[3][3] + "\t" + (int)cm.matrix[3][4] + "\t" + (int)cm.matrix[3][5] + "\t"); 
		System.out.println("row SUM = " + sum(cm.matrix[3])); 
		
		System.out.print("   starfish\t"); 
		System.out.print((int)cm.matrix[4][0] + "\t" + (int)cm.matrix[4][1] + "\t" + (int)cm.matrix[4][2] + "\t"); 
		System.out.print((int)cm.matrix[4][3] + "\t" + (int)cm.matrix[4][4] + "\t" + (int)cm.matrix[4][5] + "\t"); 
		System.out.println("row SUM = " + sum(cm.matrix[4])); 
		
		System.out.print("      watch\t"); 
		System.out.print((int)cm.matrix[5][0] + "\t" + (int)cm.matrix[5][1] + "\t" + (int)cm.matrix[5][2] + "\t"); 
		System.out.print((int)cm.matrix[5][3] + "\t" + (int)cm.matrix[5][4] + "\t" + (int)cm.matrix[5][5] + "\t"); 
		System.out.println("row SUM = " + sum(cm.matrix[5])); 
		
		System.out.println("Predicted Category"); 
		
		Mat cmT = cm.T(); 
		System.out.print(" column SUM\t");
		System.out.print(sum(cmT.matrix[0]) + "\t" + sum(cmT.matrix[1]) + "\t" + sum(cmT.matrix[2]) + "\t"); 
		System.out.println(sum(cmT.matrix[3]) + "\t" + sum(cmT.matrix[4]) + "\t" + sum(cmT.matrix[5]) + "\t"); 
		
		int cor = 0; 
		for (int i = 0; i < 6; i++) {
			cor = cor + (int)cm.matrix[i][i]; 
		}
		int total = (int)Mat.sum(cm); 
		System.out.println("total examples = " + total + ", errors = " + (total - cor) + "(" + (100 * (double)(total - cor) / (double)total) + "%)"); 
	}
	
	private static int sum(double[] in) {
		int out = 0; 
		for (int i = 0; i < in.length; i++) {
			out = out + (int)in[i]; 
		}
		return out; 
	}
	
	private static void load(Data myset, Dataset dataset) {
		
		List<Instance> trainInstList = dataset.getImages(); 
		for (int i = 0;  i < trainInstList.size(); i++) {
			Instance inst = trainInstList.get(i); 
			
			List<int[][]> feat = new ArrayList<int[][]>(); 
			if (useRGB) {
				feat.add(inst.getRedChannel()); 
				feat.add(inst.getGreenChannel()); 
				feat.add(inst.getBlueChannel()); 
				feat.add(inst.getGrayImage()); 
			} else {
				feat.add(inst.getGrayImage()); 
			}
				
			DataContainer instData = convertData(feat, 6, convertCategoryStringToEnum(inst.getLabel()).ordinal()); 
			myset.addInstance(instData);
		}
	}
	
	private static DataContainer convertData (List<int[][]> img, int numLabel, int label) {
		List<Mat> list = new ArrayList<Mat>(); 
		
		for (int i = 0; i < img.size(); i++) {
			int val[][] = img.get(i); 
			Mat imgDouble = new Mat(val.length, val[0].length); 
			for (int n = 0; n < val.length; n++) {
				for (int m = 0; m < val[0].length; m++) {
					imgDouble.matrix[n][m] = (double)val[n][m]/255*2 - 1; 
				}
			}
			//System.out.println(imgDouble);
			list.add(imgDouble); 
		}
		
		DataContainer data = new DataContainer(list); 
		Mat cl = new Mat(1, numLabel, 0); 
		cl.matrix[0][label] = 1; 
		
		data.setLabel(cl); 
		
		return data; 
	}
	
	private static Dataset moreInstances (Dataset trainset, int totalSize) {
		List<Instance> insts = trainset.getImages(); 
		List<List<Instance>> instSeq = new ArrayList<List<Instance>>(); 
		for (int i = 0; i < 6; i++) {
			instSeq.add(new ArrayList<Instance>()); 
		}
		for (int i = 0; i < insts.size(); i++) {
			instSeq.get(convertCategoryStringToEnum(insts.get(i).getLabel()).ordinal()).add(insts.get(i)); 
		}
		
		Random rnd = new Random(20); 
		for (int i = 0; i < 6; i++) {
			List<Instance> list = instSeq.get(i);
			int maxIdx = list.size(); 

			for (int idx = 0; idx < maxIdx; idx++) {
				list.add(list.get(idx).flipImageLeftToRight()); 
				list.add(list.get(idx).flipImageTopToBottom()); 
				list.add(list.get(idx).flipImageTopToBottom().flipImageLeftToRight()); 
			}
			
			maxIdx = list.size(); 
			
			while (list.size() < (totalSize / 6)) {
				for (int j = 0; j < maxIdx; j++) {
					if (rnd.nextBoolean()) {
						list.add(list.get(j).rotateImageThisManyDegrees(rnd.nextInt(21) - 10)); 
					} else {
						list.add(list.get(j).shiftImage(rnd.nextInt(11) - 5, rnd.nextInt(11) - 5)); 
					}
				}
			}
		}
		
		Dataset set = new Dataset(); 
		for (int i = 0; i < 6; i++) {
			List<Instance> list = instSeq.get(i);
			for (int j = 0; j < list.size(); j++) {
				set.add(list.get(j));
			}
		}
		
		return set; 
	}

	////////////////////////////////////////////////////////////////////////////////////////////////

}



interface ANNLayer {
	public DataContainer forwardProp(DataContainer forwardPropData, boolean training); 
	public DataContainer backProp(DataContainer backPropError, double learningRate); 
	public void setBatchSize(int size); 
	public ANNLayer copy(); 
}

class DataContainer {
	private List<Mat> imgs; 
	private Mat vector; 
	private Mat label = null; 
	
	public DataContainer(Mat vector) {
		this.vector = vector; 
		imgs = null; 
	}
	public DataContainer(List<Mat> imgs) {
		this.imgs = imgs; 
		vector = null; 
	}
	
	public List<Mat> getImgs() {
		if (imgs == null) {
			throw new RuntimeException("Error: Trying to get image list"); 
		}
		return imgs; 
	}
	public Mat getVec() {
		if (vector == null) {
			throw new RuntimeException("Error: Trying to get vector"); 
		}
		return vector; 
	}
	
	public void setLabel(Mat rowVec) {
		this.label = rowVec; 
	}
	public Mat getLabel() {
		if (label == null) {
			throw new RuntimeException("This container does not contain label"); 
		}
		return label; 
	}
}


class ConvolutionLayer implements ANNLayer {
	private Mat[][] filters; 
	private Mat[] lastInput; 
	private int numOutPla, filterSize; 
	private boolean set = false; 
	
	private int batchSize = 1; 
	private int batchCounter = 0; 
	private Mat[][] filtersUpdate; 
	
	public ConvolutionLayer(int filterSize, int numOutPla) {
		this.numOutPla = numOutPla; 
		this.filterSize = filterSize; 
	}
	
	@Override
	public DataContainer forwardProp(DataContainer forwardPropData, boolean training) {
		// TODO Auto-generated method stub
		List<Mat> input = forwardPropData.getImgs(); 
		
		if (!set) {
			filters = new Mat[this.numOutPla][input.size() + 1]; 
			filtersUpdate = new Mat[this.numOutPla][input.size() + 1]; 
			for (int i = 0; i < filters.length; i++) {
				for (int j = 0; j <filters[0].length; j++) {
					filters[i][j] = Mat.randomMatrix(filterSize, filterSize, - 0.03, 0.03); 
					filtersUpdate[i][j] = new Mat (filterSize, filterSize, 0); 
				}
			}
			lastInput = new Mat[input.size() + 1]; 
			lastInput[input.size()] = new Mat(input.get(0).size[0], input.get(0).size[1], 1); 
			set = true; 
		}
		
		for (int i = 0; i < input.size(); i++) {
			lastInput[i] = input.get(i);  
		}
		
		List<Mat> output = new ArrayList<Mat>(); 
		for (int i = 0; i < filters.length; i++) {
			Mat out = Conv.Conv2Valid(lastInput[0], filters[i][0]); 
			for (int j = 1; j < filters[i].length; j++) {
				out = out.add(Conv.Conv2Valid(lastInput[j], filters[i][j])); 
			}
			output.add(out); 
		}
		
		return new DataContainer(output); 
	}



	@Override
	public DataContainer backProp(DataContainer backPropError, double learningRate) {
		// TODO Auto-generated method stub
		List<Mat> errorList = backPropError.getImgs(); 
		
		List<Mat> outputList = new ArrayList<Mat>(); 
		for (int i = 0; i < lastInput.length - 1; i++) {
			Mat error = Conv.Conv2Full(errorList.get(0), ROT180(filters[0][i])); 
			for (int j = 1; j < filters.length; j++) {
				error = error.add(Conv.Conv2Full(errorList.get(j), ROT180(filters[j][i]))); 
			}
			outputList.add(error); 
		}
		
		for (int i = 0; i < errorList.size(); i++) {
			Mat error = errorList.get(i); 
			for (int j = 0; j < lastInput.length; j++) {
				Mat dW = Conv.Conv2Valid(ROT180(lastInput[j]), error); 
				//filters[i][j] = filters[i][j].add(dW.scale(-learningRate)); 
				filtersUpdate[i][j] = filtersUpdate[i][j].add(dW); 
			}
		}
		
		batchCounter++; 
		
		if (batchCounter >= batchSize) {
			for (int i = 0; i < filters.length; i++) {
				for (int j = 0; j <filters[0].length; j++) {
					filters[i][j] = filters[i][j].add(filtersUpdate[i][j].scale(- learningRate / batchSize)); 
					filtersUpdate[i][j] = new Mat (filterSize, filterSize, 0); 
				}
			}
			batchCounter = 0; 
		}
		
		return new DataContainer(outputList); 
	}

	@Override
	public ANNLayer copy() {
		// TODO Auto-generated method stub
		ConvolutionLayer layer = new ConvolutionLayer(filterSize, numOutPla); 
		layer.filters = new Mat[this.filters.length][this.filters[0].length]; 
		for (int i = 0; i < layer.filters.length; i++) {
			for (int j = 0; j < layer.filters[i].length; j++) {
				layer.filters[i][j] = this.filters[i][j].copy(); 
			}
		}
		layer.set = this.set; 
		
		layer.lastInput = new Mat[this.lastInput.length]; 
		for (int i = 0; i < layer.lastInput.length; i++) {
			layer.lastInput[i] = this.lastInput[i].copy(); 
		}
		
		return layer; 
	}
	
	private Mat ROT180(Mat input) {
		Mat output = new Mat(input.size[0], input.size[1], 0); 
		for (int i = 0; i < input.size[0]; i++) {
			for (int j = 0; j < input.size[1]; j++) {
				output.matrix[i][j] = input.matrix[input.size[0]-i-1][input.size[1]-j-1]; 
			}
		}
		return output; 
	}

	@Override
	public void setBatchSize(int size) {
		// TODO Auto-generated method stub
		if (size <= 0) {
			throw new IllegalArgumentException(); 
		}
		this.batchSize = size; 
	}
}

/**
 * Finalized Class
 * DO NOT CHANGE THIS CLASS
 * Author: Zhanpeng Zeng
 */
class Conv {
	public static Mat Conv2Full(Mat input, Mat kernel) {
		int inputH = input.size[0]; 
		int inputW = input.size[1]; 
		int kernH = kernel.size[0]; 
		int kernW = kernel.size[1]; 
		
		Mat output = new Mat(inputH + kernH - 1, inputW + kernW - 1, 0); 
		
		for (int i = 0; i < inputH; i++) {
			for (int j = 0; j < inputW; j++) {
				Mat temp = kernel.scale(input.matrix[i][j]); 
				for (int m = 0; m < kernH; m++) {
					for (int n = 0; n < kernW; n++) {
						output.matrix[i+m][j+n] += temp.matrix[m][n]; 
					}
				}
			}
		}
		return output; 
	}
	
	public static Mat Conv2Valid(Mat input, Mat kernel) {
		int inputH = input.size[0]; 
		int inputW = input.size[1]; 
		int kernH = kernel.size[0]; 
		int kernW = kernel.size[1]; 
		if (inputH - kernH + 1 <= 0 || inputW - kernW + 1 <= 0) {
			throw new RuntimeException("invalid input size"); 
		}
		
		Mat fullOutput = Conv2Full(input, kernel); 
		
		Mat output = new Mat(inputH - kernH + 1, inputW - kernW + 1); 
		for (int i = 0; i < output.size[0]; i++) {
			for (int j = 0; j < output.size[1]; j++) {
				output.matrix[i][j] = fullOutput.matrix[i + kernH - 1][j + kernW - 1]; 
			}
		}
		return output; 
	}
	
	public static Mat Conv2Same(Mat input, Mat kernel) {
		int inputH = input.size[0]; 
		int inputW = input.size[1]; 
		int kernH = kernel.size[0]; 
		int kernW = kernel.size[1]; 
		if (inputH - kernH + 1 <= 0 || inputW - kernW + 1 <= 0) {
			throw new RuntimeException("invalid input size"); 
		}
		
		Mat fullOutput = Conv2Full(input, kernel); 
		
		Mat output = new Mat(inputH, inputW); 
		for (int i = 0; i < output.size[0]; i++) {
			for (int j = 0; j < output.size[1]; j++) {
				output.matrix[i][j] = fullOutput.matrix[i + (kernH - 1)/2][j + (kernW - 1)/2]; 
			}
		}
		return output; 
	}
}


class Data {
	private List<DataContainer> instance; 
	private int numLabels; 
	private Random rnd = new Random(1000); 
	
	public Data() {
		instance = new ArrayList<DataContainer>(); 
	}
	
	public int size() {
		return instance.size(); 
	}
	
	public int labelSize() {
		return numLabels;
	}
	
	public DataContainer get(int idx) {
		return instance.get(idx); 
	}
	
	public void shuffle() {
		Collections.shuffle(instance, rnd);
	}
	
	public void add(Mat instance) {
		this.instance.add(new DataContainer(instance)); 
	}
	
	public void loadData(Vector<Vector<Double>> dataVectors, int numLabels) {
		for (int i = 0; i < dataVectors.size(); i++) {
			Mat inst = new Mat(1, dataVectors.get(0).size()-1); 
			for (int j = 0; j < inst.size[1]; j++) {
				inst.matrix[0][j] = dataVectors.get(i).get(j); 
			}
			Mat label = new Mat(1, numLabels, 0); 
			double idx = dataVectors.get(i).get(inst.size[1]); 
			label.matrix[0][(int)idx] = 1; 
			DataContainer result = new DataContainer(inst); 
			result.setLabel(label);
			this.instance.add(result); 
			this.numLabels = numLabels; 
		}
	}
	
	public void addInstance(DataContainer inst) {
		instance.add(inst); 
		numLabels = inst.getLabel().size[1]; 
	}
	
	public Data subset(double fraction) {
		int numInst = (int)(((double)this.instance.size()) * fraction);
		Data subset = new Data(); 
		for (int i = 0; i < this.instance.size(); i++) {
			subset.addInstance(instance.get(i));
		}
		while (subset.size() > numInst) {
			subset.instance.remove(rnd.nextInt(subset.size())); 
		}
		return subset; 
	}
	
}


class FullyConnectedLayer implements ANNLayer{

	private Mat inputWeight; 
	private boolean set = false; 
	private int fout; 
	
	private int batchSize = 1; 
	private int batchCounter = 0; 
	private Mat weightUpdate; 
	
	private Mat lastInputBiased = null; 
	
	public FullyConnectedLayer(int fout) {
		this.fout = fout; 
	}
	
	public FullyConnectedLayer() {
		
	}
	
	@Override
	public DataContainer forwardProp(DataContainer forwardPropData, boolean training) {
		// TODO Auto-generated method stub
		Mat input = forwardPropData.getVec(); 
		
		if (!set) {
			inputWeight = Mat.randomMatrix(input.size[1] + 1, fout, - 0.3, 0.3); 
			weightUpdate = new Mat(input.size[1] + 1, fout, 0); 
			set = true; 
		}
		
		Mat inputBiased = (input).appendRight(new Mat(1,1,1)); 
		Mat output = inputBiased.mul(inputWeight); 
		if (training) {
			lastInputBiased = inputBiased; 
		}
		else {
			lastInputBiased = null; 
		}
		
		return new DataContainer(output); 
	}
	
	@Override
	public DataContainer backProp(DataContainer backPropError, double learningRate) {
		// TODO Auto-generated method stub
		if (lastInputBiased == null) {
			throw new RuntimeException("Invalid operation"); 
		}
		Mat error = backPropError.getVec(); 
		
		Mat dW = lastInputBiased.T().mul(error); 
		weightUpdate = weightUpdate.add(dW); 
		//inputWeight = inputWeight.add(dW.scale(learningRate)); 
		batchCounter++; 
		
		Mat temp = error.mul(inputWeight.T()); 
        Mat output = new Mat(temp.size[0], temp.size[1] - 1); 
        for (int i = 0; i < output.size[1]; i++) {
        		output.matrix[0][i] = temp.matrix[0][i]; 
        }
        
        if (batchCounter >= batchSize) {
        		inputWeight = inputWeight.add(weightUpdate.scale( - learningRate / batchSize)); 
        		weightUpdate = new Mat(inputWeight.size[0], inputWeight.size[1], 0); 
			batchCounter = 0; 
		}
		
		return new DataContainer(output); 
	}

	@Override
	public ANNLayer copy() {
		// TODO Auto-generated method stub
		FullyConnectedLayer output = new FullyConnectedLayer(); 
		output.inputWeight = this.inputWeight.copy(); 
		output.fout = this.fout; 
		output.set = this.set; 
		return output; 
	}

	@Override
	public void setBatchSize(int size) {
		// TODO Auto-generated method stub
		if (size <= 0) {
			throw new IllegalArgumentException(); 
		}
		this.batchSize = size; 
	}
	
	

}


class LinearActivationLayer implements ANNLayer {

	private boolean set; 
	private double dropOutRate, dropOutCorrection; 
	private Random rnd = new Random(3000); 
	private Mat dropOutMask; 
	
	public LinearActivationLayer(double dropOutRate) {
		this.dropOutRate = dropOutRate; 
	}
	
	@Override
	public DataContainer forwardProp(DataContainer forwardPropData, boolean training) {
		// TODO Auto-generated method stub
		Mat input = forwardPropData.getVec(); 
		
		if (!set) {
			dropOutMask = new Mat(1, input.size[1], 1); 
			for (int i = 0; i < (int)((double)input.size[1] * dropOutRate); i++) {
				dropOutMask.matrix[0][i] = 0; 
			}
			this.dropOutCorrection = 1 - ((int)((double)input.size[1] * dropOutRate)) / ((double) input.size[1]); 
			set = true; 
		}
		
		if (training) {
			shuffleDropOutMask(); 
			return new DataContainer(input.eleWiseMul(dropOutMask)); 
		} else {
			return new DataContainer(input.scale(dropOutCorrection)); 
		}
	}

	@Override
	public DataContainer backProp(DataContainer backPropError, double learningRate) {
		// TODO Auto-generated method stub
		
		Mat error = backPropError.getVec(); 
		Mat output = dropOutMask.eleWiseMul(error); 
		
		return new DataContainer(output);
	}

	@Override
	public ANNLayer copy() {
		// TODO Auto-generated method stub
		return this; 
	}
	
	private void shuffleDropOutMask() {
		double swap; 
		for (int i = dropOutMask.size[1] - 1; i >= 1; i--) {
			int rndIdx = rnd.nextInt(i+1); 
			swap = dropOutMask.matrix[0][rndIdx]; 
			dropOutMask.matrix[0][rndIdx] = dropOutMask.matrix[0][i]; 
			dropOutMask.matrix[0][i] = swap; 
		}
    }

	@Override
	public void setBatchSize(int size) {
		// TODO Auto-generated method stub
		
	}
}


class LinearActivationLayer2D implements ANNLayer {

	private boolean set; 
	private double dropOutRate, dropOutCorrection; 
	private Mat dropOutMask[]; 
	private int maskSizeH, maskSizeW; 
	
	public LinearActivationLayer2D(double dropOutRate) {
		this.dropOutRate = dropOutRate; 
		this.dropOutCorrection = 1 - dropOutRate; 
	}
	
	@Override
	public DataContainer forwardProp(DataContainer forwardPropData, boolean training) {
		// TODO Auto-generated method stub
		List<Mat> input = forwardPropData.getImgs(); 
		
		if (!set) {
			dropOutMask = new Mat[input.size()]; 
			maskSizeH = input.get(0).size[0]; 
			maskSizeW = input.get(0).size[1];
			set = true; 
		}
		
		List<Mat> output = new ArrayList<Mat>(); 
		if (training) {
			shuffleDropOutMask(); 
			for (int i = 0; i < input.size(); i++) {
				output.add(input.get(i).eleWiseMul(dropOutMask[i])); 
			}
		} else {
			for (int i = 0; i < input.size(); i++) {
				output.add(input.get(i).scale(this.dropOutCorrection)); 
			}
		}
		return new DataContainer(output); 
	}

	@Override
	public DataContainer backProp(DataContainer backPropError, double learningRate) {
		// TODO Auto-generated method stub
		
		List<Mat> error = backPropError.getImgs(); 
		
		List<Mat> output = new ArrayList<Mat>(); 
		for (int i = 0; i < error.size(); i++) {
			output.add(error.get(i).eleWiseMul(dropOutMask[i])); 
		}
		
		return new DataContainer(output);
	}

	@Override
	public ANNLayer copy() {
		// TODO Auto-generated method stub
		return this; 
	}
	
	private void shuffleDropOutMask() {
		for (int i = 0; i < this.dropOutMask.length; i++) {
			dropOutMask[i] = Mat.randomMatrix(maskSizeH, maskSizeW, 0, 1); 
			for (int a = 0; a < maskSizeH; a++) {
				for (int b = 0; b < maskSizeW; b++) {
					if (dropOutMask[i].matrix[a][b] < this.dropOutRate) {
						dropOutMask[i].matrix[a][b] = 0; 
					}
					else {
						dropOutMask[i].matrix[a][b] = 1; 
					}
				}
			}
		}
    }

	@Override
	public void setBatchSize(int size) {
		// TODO Auto-generated method stub
		
	}
}


class Mat {
	private static Random rnd = new Random(500000); 
	
	public double matrix[][]; 
	public int size[]; 
	
	public Mat(int a, int b, double value) {
		size = new int[]{a, b}; 
		
		matrix = new double[a][b]; 
		for (int i = 0; i < a; i++) {
			for (int j = 0; j < b; j++)
				matrix[i][j] = value; 
		}
	}
	
	public Mat(double[] rowVector) {
		size = new int[]{1, rowVector.length}; 
		this.matrix = new double[1][rowVector.length]; 
		for (int i = 0; i < rowVector.length; i++) {
			matrix[0][i] = rowVector[i]; 
		}
	}
	
	public Mat(double[][] matrix) {
		size = new int[]{matrix.length, matrix[0].length}; 
		this.matrix = matrix; 
	}
	
	public Mat(int a, int b) {
		size = new int[]{a, b}; 
		matrix = new double[a][b]; 
	}
	
	public static Mat randomMatrix(int a, int b, double min, double max) {
		double scale = (max - min); 
		double offset = min; 
		
		Mat result = new Mat(a, b, 0); 
		for (int i = 0; i < a; i++) {
			for (int j = 0; j < b; j++) {
				result.matrix[i][j] = rnd.nextDouble() * scale + offset; 
			}
		}
		
		return result; 
	}
	
	public Mat range(int aMin, int aMax, int bMin, int bMax) {
		if (aMin < 0 || aMax >= size[0] || bMin < 0 || bMax >= size[1]) {
			throw new IllegalArgumentException("illegal range"); 
		}
		Mat result = new Mat(aMax - aMin + 1, bMax - bMin + 1, 0); 
		for (int i = aMin; i <= aMax; i++) {
			for (int j = bMin; j <= bMax; j++) {
				result.matrix[aMin-i][bMin-j] = this.matrix[i][j]; 
			}
		}
		return result; 
	}
	
	public static double sum(Mat matrix) {
		double result = 0; 
		for (int i = 0; i < matrix.size[0]; i++) {
			for (int j = 0; j < matrix.size[1]; j++) {
				result = result + matrix.matrix[i][j]; 
			}
		}
		return result; 
	}
	
	public static Result max(Mat matrix) {
		Result output = new Result(); 
		output.max = matrix.matrix[0][0]; 
		output.idx[0] = 0; 
		output.idx[1] = 0; 
		
		for (int i = 0; i < matrix.size[0]; i++) {
			for (int j = 0; j < matrix.size[1]; j++) {
				if (output.max < matrix.matrix[i][j]) {
					output.max = matrix.matrix[i][j]; 
					output.idx[0] = i; 
					output.idx[1] = j; 
				}
			}
		}
		return output; 
	}
	
	public Mat appendRight(Mat m) {
		List<Mat> list = new ArrayList<Mat>(); 
		list.add(this); 
		list.add(m); 
		return Mat.marge(list); 
	}
	
	public static Mat marge(List<Mat> list) {
		int height = list.get(0).size[0]; 
		for (Iterator<Mat> itr = list.iterator(); itr.hasNext(); ) {
			if (itr.next().size[0] != height)
				throw new IllegalArgumentException(); 
		}
		
		int width = 0; 
		for (Iterator<Mat> itr = list.iterator(); itr.hasNext(); ) 
			width = width + itr.next().size[1]; 
		
		Mat result = new Mat(height, width, 0); 
		for (int i = 0; i < height; i++) {
			int k = 0; 
			for (Iterator<Mat> itr = list.iterator(); itr.hasNext(); ) {
				Mat temp = itr.next(); 
				for (int j = 0; j < temp.size[1]; j++) {
					result.matrix[i][k] = temp.matrix[i][j]; 
					k++; 
				}
			}
		}
		
		return result; 
	}
	
	public Mat copy() {
		Mat result = new Mat(this.size[0], this.size[1], 0); 
		for (int i = 0; i < result.size[0]; i++) {
			for (int j = 0; j < result.size[1]; j++) {
				result.matrix[i][j] = this.matrix[i][j]; 
			}
		}
		return result; 
	}
	
	public Mat eleWiseMul(Mat B) {
		if ((B.size[0] != this.size[0]) ||(B.size[1] != this.size[1])) {
			throw new IllegalArgumentException("Dimension mismatch"); 
		}
		Mat result = new Mat(this.size[0], this.size[1]); 
		for (int i = 0; i < result.size[0]; i++) {
			for (int j = 0; j < result.size[1]; j++) {
				result.matrix[i][j] = this.matrix[i][j] * B.matrix[i][j]; 
			}
		}
		return result; 
	}
	
	public Mat mul(Mat B) {
		if (this.size[1] != B.size[0]) {
			throw new IllegalArgumentException(); 
		}
		Mat result = new Mat(this.size[0], B.size[1]); 
		
		for (int i = 0; i < result.size[0]; i++) {
			for (int j = 0; j < result.size[1]; j++) {
				double accumulator = 0; 
				for (int k = 0; k < this.size[1]; k++) {
					accumulator = accumulator + this.matrix[i][k] * B.matrix[k][j]; 
				}
				result.matrix[i][j] = accumulator; 
			}
		}
		
		return result; 
	}
	
	public Mat add(Mat B) {
		if ((this.size[0] != B.size[0]) || (this.size[1] != B.size[1])) {
			throw new IllegalArgumentException(); 
		}
		Mat result = new Mat(this.size[0], this.size[1]); 
		
		for (int i = 0; i < result.size[0]; i++) {
			for (int j = 0; j < result.size[1]; j++) {
				result.matrix[i][j] = this.matrix[i][j] + B.matrix[i][j]; 
			}
		}
		return result; 
	}
	
	public Mat scale(double scale) {
		Mat result = new Mat(this.size[0], this.size[1]); 
		for (int i = 0; i < result.size[0]; i++) {
			for (int j = 0; j < result.size[1]; j++)
				result.matrix[i][j] = this.matrix[i][j] * scale; 
		}
		return result; 
	}
	
	public Mat T() {
		Mat result = new Mat(this.size[1], this.size[0]); 
		for (int i = 0; i < result.size[0]; i++) {
			for (int j = 0; j < result.size[1]; j++)
				result.matrix[i][j] = this.matrix[j][i]; 
		}
		return result; 
	}
	
	public static Mat pow(Mat input, double power) {
		Mat result = new Mat(input.size[0], input.size[1]); 
		for (int i = 0; i < result.size[0]; i++) {
			for (int j = 0; j < result.size[1]; j++) {
				result.matrix[i][j] = Math.pow(input.matrix[i][j], power); 
			}
		}
        return result;
	}
	
	public Mat row(int idx) {
		return new Mat(this.matrix[idx]); 
	}
	
	public String toString() {
		String result = "[";
		for (int i = 0; i < this.size[0]; i++) {
			for (int j = 0; j < this.size[1]; j++) {
				result = result + String.format("%.4f", this.matrix[i][j]);
				if (j != this.size[1] - 1)
					result = result + ", ";
			}
			
			if (i != this.size[0] - 1)
				result = result + "; \n"; 
		}
		result = result +"];\n";
		return result; 
	}

}

class Result {
	public double max; 
	public int idx[] = new int[2]; 
}



class MaxPoolingLayer implements ANNLayer{
	private boolean set = false; 
	private Mat[] poolingMask; 
	
	public MaxPoolingLayer() {
		
	}
	
	@Override
	public DataContainer forwardProp(DataContainer forwardPropData, boolean training) {
		// TODO Auto-generated method stub

		List<Mat> input = forwardPropData.getImgs(); 
		if (!set) {
			poolingMask = new Mat[input.size()]; 
			set = true; 
		}
		
		List<Mat> output = new ArrayList<Mat>(); 
		for (int i = 0; i < input.size(); i++) {
			PoolingResult result = maxPooling(input.get(i)); 
			output.add(result.output); 
			poolingMask[i] = result.mask; 
		}
		
		return new DataContainer(output); 
	}

	@Override
	public DataContainer backProp(DataContainer backPropError, double learningRate) {
		// TODO Auto-generated method stub

		List<Mat> input = backPropError.getImgs(); 
		
		List<Mat> output = new ArrayList<Mat>(); 
		for (int i = 0; i < input.size(); i++) {
			output.add(upsample(input.get(i), i)); 
		}
		return new DataContainer(output); 
	}
	
	private PoolingResult maxPooling(Mat input) {
		Mat minput; 
		if (input.size[0] % 2 == 0 && input.size[1] % 2 == 0) {
			minput = input; 
		} else {
			minput = new Mat (input.size[0] + (input.size[0] % 2), input.size[1] + (input.size[1] % 2), - Double.MAX_VALUE); 
			for (int i = 0; i < input.size[0]; i++) {
				for (int j = 0; j < input.size[1]; j++) {
					minput.matrix[i][j] = input.matrix[i][j]; 
				}
			}
		}
		
		Mat output = new Mat(minput.size[0] / 2, minput.size[1] / 2); 
		Mat mask = new Mat(input.size[0], input.size[1], 0); 
		for (int i = 0; i < minput.size[0] ; i = i + 2) {
			for (int j = 0; j < minput.size[1]; j = j + 2) {
				Mat temp = new Mat(2,2); 
				temp.matrix[0][0] = minput.matrix[i][j];
				temp.matrix[0][1] = minput.matrix[i][j+1];
				temp.matrix[1][0] = minput.matrix[i+1][j];
				temp.matrix[1][1] = minput.matrix[i+1][j+1];
				Result max = Mat.max(temp); 
				output.matrix[i/2][j/2] = max.max; 
				mask.matrix[i+max.idx[0]][j+max.idx[1]] = 1; 
			}
		}
		
		return new PoolingResult(output, mask); 
	}
	
	private Mat upsample(Mat input, int maskIdx) {
		Mat output = new Mat(poolingMask[maskIdx].size[0], poolingMask[maskIdx].size[1], 0); 
		
		for (int i = 0; i < output.size[0]; i++) {
			for (int j = 0; j < output.size[1]; j++) {
				if (poolingMask[maskIdx].matrix[i][j] == 1) {
					output.matrix[i][j] = input.matrix[i/2][j/2];
				}
			}
		}
		
		return output; 
	}
	
	@Override
	public ANNLayer copy() {
		// TODO Auto-generated method stub
		return this; 
	}

	@Override
	public void setBatchSize(int size) {
		// TODO Auto-generated method stub
		
	}

}

class PoolingResult {
	public Mat output; 
	public Mat mask; 
	public PoolingResult(Mat output, Mat mask) {
		this.output = output; 
		this.mask = mask; 
	}
}


class NeuralNet {
	private List<ANNLayer> currNetwork; 
	
	private Data trainingData, tuningData; 
	
	public double trainLoss; 
	public double trainError, tuneError; 
	
	private double bestTuneError; 
	private List<ANNLayer> earlyStopNetwork; 
	
	private boolean networkStructureFinalized = false; 
	
	public NeuralNet(Data trainingData, Data tuningData) {
		currNetwork = new ArrayList<ANNLayer>(); 
		this.trainingData = trainingData; 
		this.tuningData = tuningData; 
	}
	
	public void setBatchSize(int size) {
		for (int i = 0; i < currNetwork.size(); i++) {
			currNetwork.get(i).setBatchSize(size);
		}
	}
	
	public void train(double learningRate, boolean suffleTrainingSet) {
		if (!networkStructureFinalized) {
			throw new RuntimeException("Network Configuration not finalized"); 
		}
		if (suffleTrainingSet) {
			trainingData.shuffle();
		}
		
		for (int idx = 0; idx < this.trainingData.size(); idx++) {
			DataContainer inst = trainingData.get(idx); 
			DataContainer forwardPropData = inst; 
			for (int layer = 0; layer < this.currNetwork.size(); layer++) {
				ANNLayer currLayer = currNetwork.get(layer); 
				forwardPropData = currLayer.forwardProp(forwardPropData, true); 
			}
			
			//---------------------------------------
			Mat output = forwardPropData.getVec(); 
			Mat y = inst.getLabel(); 
			DataContainer backPropError = new DataContainer(output.add(y.scale(-1))); 
			//---------------------------------------
			
			for (int layer = this.currNetwork.size() - 1; layer >= 0; layer--) {
				ANNLayer currLayer = currNetwork.get(layer); 
				backPropError = currLayer.backProp(backPropError, learningRate); 
			}
		}
		
		error(); 
		if (tuneError < bestTuneError) {
			bestTuneError = tuneError; 
			copyCurrNetworkAsEarlyStopNetwork(); 
		}
	}
	
	
	private void error() {
		trainLoss = 0; 
		trainError = 0; 
		tuneError = 0; 

		for (int idx = 0; idx < this.trainingData.size(); idx++) {
			DataContainer inst = trainingData.get(idx); 
			DataContainer forwardPropData = inst; 
			for (int layer = 0; layer < this.currNetwork.size(); layer++) {
				ANNLayer currLayer = currNetwork.get(layer); 
				forwardPropData = currLayer.forwardProp(forwardPropData, false); 
			}
			
			Mat output = forwardPropData.getVec(); 
			Result maxOutput = Mat.max(output); 
			Result actual = Mat.max(inst.getLabel()); 
			if (actual.idx[1] != maxOutput.idx[1]) {
				trainError++; 
			}
			
			Mat y = inst.getLabel(); 
			trainLoss = trainLoss + 0.5 * Mat.sum(Mat.pow(y.add(output.scale(-1)), 2)); 
		}
		trainError = trainError / trainingData.size(); 
		
		for (int idx = 0; idx < this.tuningData.size(); idx++) {
			DataContainer inst = tuningData.get(idx); 
			DataContainer forwardPropData = inst; 
			for (int layer = 0; layer < this.currNetwork.size(); layer++) {
				ANNLayer currLayer = currNetwork.get(layer); 
				forwardPropData = currLayer.forwardProp(forwardPropData, false); 
			}
			
			Mat output = forwardPropData.getVec(); 
			Result maxOutput = Mat.max(output); 
			Result actual = Mat.max(inst.getLabel()); 
			if (actual.idx[1] != maxOutput.idx[1]) {
				tuneError++; 
			}
		}
		tuneError = tuneError / tuningData.size(); 
		
	}
	
	public Mat predict(DataContainer inst) {
		DataContainer forwardPropData = inst; 
		for (int layer = 0; layer < this.currNetwork.size(); layer++) {
			ANNLayer currLayer = currNetwork.get(layer); 
			forwardPropData = currLayer.forwardProp(forwardPropData, false); 
		}
		
		return forwardPropData.getVec(); 
	}
	
	public NeuralNet getEarlyStopModel() {
		NeuralNet model = new NeuralNet(trainingData, tuningData); 
		model.currNetwork = this.earlyStopNetwork; 
		return model; 
	}
	
	private void copyCurrNetworkAsEarlyStopNetwork() {
		earlyStopNetwork = new ArrayList<ANNLayer>(); 
		for (int i = 0; i < this.currNetwork.size(); i++) {
			earlyStopNetwork.add(currNetwork.get(i).copy()); 
		}
	}
	
	public void finalizeConfiguration() {
		networkStructureFinalized = true; 
		
		error(); 
		
		bestTuneError = tuneError; 
		copyCurrNetworkAsEarlyStopNetwork(); 
	}
	
	public void stuckConvolutionLayer(int filerSize, int numOutPla) {
		currNetwork.add(new ConvolutionLayer(filerSize, numOutPla)); 
	}
	
	public void stuckMaxPoolingLayer() {
		currNetwork.add(new MaxPoolingLayer()); 
	}
	
	public void stuckVectorizationLayer() {
		currNetwork.add(new VectorizationLayer()); 
	}
	
	public void stuckFullyConnectedLayer(int fout) {
		currNetwork.add(new FullyConnectedLayer(fout)); 
	}
	
	public void stuckReLUActivationLayer(double dropOutRate) {
		currNetwork.add(new ReLUActivationLayer(dropOutRate)); 
	}
	
	public void stuckLinearActivationLayer(double dropOutRate) {
		currNetwork.add(new LinearActivationLayer(dropOutRate)); 
	}
	
	public void stuckSigmoidActivationLayer(double dropOutRate) {
		currNetwork.add(new SigmoidActivationLayer(dropOutRate)); 
	}
	
	public void stuckLinearActivationLayer2D(double dropOutRate) {
		currNetwork.add(new LinearActivationLayer2D(dropOutRate)); 
	}
}


class PerformanceAnalyser {
	
	List<Record> records; 
	
	public PerformanceAnalyser() {
		records = new ArrayList<Record>(); 
	}
	
	public static Record measure(NeuralNet model, Data test, boolean disp) {
		Record rc = new Record(); 
		rc.trainLoss = model.trainLoss; 
		rc.trainError = model.trainError; 
		rc.tuneError = model.tuneError; 
		double testError = 0; 
		for (int i = 0; i < test.size(); i++) {
			DataContainer inst = test.get(i); 
			Result maxOutput = Mat.max(model.predict(inst)); 
			Result maxY = Mat.max(inst.getLabel()); 
			if (maxOutput.idx[1] != maxY.idx[1])
				testError++; 
		}
		rc.testError = testError/test.size(); 
		if (disp) {
			System.out.print("Train Loss: " + String.format("%.4f", rc.trainLoss)); 
			System.out.print("\tTrain Error: " + String.format("%.4f", rc.trainError)); 
			System.out.print("\tTune Error: " + String.format("%.4f", rc.tuneError)); 
			System.out.println("\tTest Error: " + String.format("%.4f", rc.testError)); 
		}
		return rc; 
	}
	
	public Mat measure(NeuralNet model, Data test, boolean disp, int epoch) {
		
		Mat matrix = new Mat(test.labelSize(),test.labelSize(), 0); 
		
		Record rc = new Record(); 
		rc.trainLoss = model.trainLoss; 
		rc.trainError = model.trainError; 
		rc.tuneError = model.tuneError; 
		
		double testError = 0; 
		for (int i = 0; i < test.size(); i++) {
			DataContainer inst = test.get(i); 
			Result maxOutput = Mat.max(model.predict(inst)); 
			Result maxY = Mat.max(inst.getLabel()); 
			
			matrix.matrix[maxOutput.idx[1]][maxY.idx[1]]++; 
			if (maxOutput.idx[1] != maxY.idx[1])
				testError++; 
		}
		rc.testError = testError/test.size(); 
		if (disp) {
			System.out.print("Train Loss: " + String.format("%.4f", rc.trainLoss)); 
			System.out.print("\tTrain Error: " + String.format("%.4f", rc.trainError)); 
			System.out.print("\tTune Error: " + String.format("%.4f", rc.tuneError)); 
			System.out.println("\tTest Error: " + String.format("%.4f", rc.testError)); 
		}
		
		rc.epoch = epoch; 
		records.add(rc); 
		
		return matrix; 
	}
	
	public static Mat confusionMatrix(NeuralNet model, Data test) {
		Mat matrix = new Mat(test.labelSize(),test.labelSize(), 0); 
		for (int instanceIdx = 0; instanceIdx < test.size(); instanceIdx++) {
			DataContainer inst = test.get(instanceIdx); 
			Result maxOutput = Mat.max(model.predict(inst)); 
			Result maxY = Mat.max(inst.getLabel()); 
			
			matrix.matrix[maxOutput.idx[1]][maxY.idx[1]]++; 
		}
		return matrix; 
	}
	
	public void printRecords(boolean includeDesciption) {
		if (includeDesciption) {
			for (int i = 0; i < records.size(); i++) {
				Record rc = records.get(i); 
				System.out.print("Epoch: " + rc.epoch); 
				System.out.print("\tTrain Loss: " + String.format("%.4f", rc.trainLoss)); 
				System.out.print("\tTrain Error: " + String.format("%.4f", rc.trainError)); 
				System.out.print("\tTune Error: " + String.format("%.4f", rc.tuneError)); 
				System.out.println("\tTest Error: " + String.format("%.4f", rc.testError)); 
			}
		} else {
			for (int i = 0; i < records.size(); i++) {
				Record rc = records.get(i); 
				System.out.print(rc.epoch); 
				System.out.print("\t" + String.format("%.4f", rc.trainLoss)); 
				System.out.print("\t" + String.format("%.4f", rc.trainError)); 
				System.out.print("\t" + String.format("%.4f", rc.tuneError)); 
				System.out.println("\t" + String.format("%.4f", rc.testError) + ";"); 
			}
		}
	}
	
	public void getROCs(NeuralNet model, Data test) {
		
	}
}

class Record {
	int epoch; 
	double trainLoss; 
	double trainError; 
	double tuneError; 
	double testError; 
}


class ReLUActivationLayer implements ANNLayer {

	private boolean set; 
	private double dropOutRate, dropOutCorrection; 
	private Random rnd = new Random(3500); 
	private Mat dropOutMask; 
	
	private Mat lastInput; 
	
	public ReLUActivationLayer(double dropOutRate) {
		this.dropOutRate = dropOutRate; 
	}
	
	@Override
	public DataContainer forwardProp(DataContainer forwardPropData, boolean training) {
		// TODO Auto-generated method stub
		Mat input = forwardPropData.getVec(); 
		
		if (!set) {
			dropOutMask = new Mat(1, input.size[1], 1); 
			for (int i = 0; i < (int)((double)input.size[1] * dropOutRate); i++) {
				dropOutMask.matrix[0][i] = 0; 
			}
			this.dropOutCorrection = 1 - ((int)((double)input.size[1] * dropOutRate)) / ((double) input.size[1]); 
			set = true; 
		}
		
		if (training) {
			shuffleDropOutMask(); 
			lastInput = input; 
			return new DataContainer(activate(input).eleWiseMul(dropOutMask)); 
		} else {
			lastInput = null; 
			return new DataContainer(activate(input).scale(dropOutCorrection)); 
		}
	}

	@Override
	public DataContainer backProp(DataContainer backPropError, double learningRate) {
		// TODO Auto-generated method stub
		if (lastInput == null) {
			throw new RuntimeException("Invalid operation"); 
		}
		
		Mat error = backPropError.getVec(); 
		Mat output = derivative(lastInput).eleWiseMul(dropOutMask).eleWiseMul(error); 
		
		return new DataContainer(output);
	}

	@Override
	public ANNLayer copy() {
		// TODO Auto-generated method stub
		return this; 
	}

	private Mat activate(Mat input) {
		Mat output = new Mat(input.size[0], input.size[1]);
		for (int i = 0; i < input.size[1]; i++) {
			if (input.matrix[0][i] <= 0)
				output.matrix[0][i] = 0.01 * input.matrix[0][i]; 
			else
				output.matrix[0][i] = input.matrix[0][i]; 
 		}
		return output; 
	}
	
	private Mat derivative(Mat input) {
		Mat result = new Mat(input.size[0], input.size[1]);
		for (int i = 0; i < input.size[1]; i++) {
			if (input.matrix[0][i] <= 0)
				result.matrix[0][i] = 0.01; 
			else
				result.matrix[0][i] = 1; 
 		}
		return result; 
	}
	
	private void shuffleDropOutMask() {
		double swap; 
		for (int i = dropOutMask.size[1] - 1; i >= 1; i--) {
			int rndIdx = rnd.nextInt(i+1); 
			swap = dropOutMask.matrix[0][rndIdx]; 
			dropOutMask.matrix[0][rndIdx] = dropOutMask.matrix[0][i]; 
			dropOutMask.matrix[0][i] = swap; 
		}
    }

	@Override
	public void setBatchSize(int size) {
		// TODO Auto-generated method stub
		
	}
}


class SigmoidActivationLayer implements ANNLayer {

	private boolean set; 
	private double dropOutRate, dropOutCorrection; 
	private Random rnd = new Random(4500); 
	private Mat dropOutMask; 
	
	private Mat lastOutputDroped; 
	
	public SigmoidActivationLayer(double dropOutRate) {
		this.dropOutRate = dropOutRate; 
	}
	
	@Override
	public DataContainer forwardProp(DataContainer forwardPropData, boolean training) {
		// TODO Auto-generated method stub
		Mat input = forwardPropData.getVec(); 
		
		if (!set) {
			dropOutMask = new Mat(1, input.size[1], 1); 
			for (int i = 0; i < (int)((double)input.size[1] * dropOutRate); i++) {
				dropOutMask.matrix[0][i] = 0; 
			}
			this.dropOutCorrection = 1 - ((int)((double)input.size[1] * dropOutRate)) / ((double) input.size[1]); 
			set = true; 
		}
		
		if (training) {
			shuffleDropOutMask(); 
			lastOutputDroped = activate(input).eleWiseMul(dropOutMask); 
			return new DataContainer(lastOutputDroped); 
		} else {
			lastOutputDroped = null; 
			return new DataContainer(activate(input).scale(dropOutCorrection)); 
		}
	}


	@Override
	public DataContainer backProp(DataContainer backPropError, double learningRate) {
		// TODO Auto-generated method stub
		if (lastOutputDroped == null) {
			throw new RuntimeException("Invalid operation"); 
		}
		
		Mat error = backPropError.getVec(); 
		Mat output = derivative(lastOutputDroped).eleWiseMul(dropOutMask).eleWiseMul(error); 
		
		return new DataContainer(output);
	}

	@Override
	public ANNLayer copy() {
		// TODO Auto-generated method stub
		return this; 
	}

	private Mat activate(Mat input) {
		Mat output = new Mat(input.size[0], input.size[1]);
		for (int i = 0; i < input.size[1]; i++) {
			output.matrix[0][i] = 1 / (1 + Math.exp(-input.matrix[0][i]));
 		}
		return output; 
	}
	
	private Mat derivative(Mat input) {
		Mat result = new Mat(input.size[0], input.size[1]);
		for (int i = 0; i < input.size[1]; i++) {
			result.matrix[0][i] = input.matrix[0][i] * (1 - input.matrix[0][i]);
 		}
		return result; 
	}
	
	private void shuffleDropOutMask() {
		double swap; 
		for (int i = dropOutMask.size[1] - 1; i >= 1; i--) {
			int rndIdx = rnd.nextInt(i+1); 
			swap = dropOutMask.matrix[0][rndIdx]; 
			dropOutMask.matrix[0][rndIdx] = dropOutMask.matrix[0][i]; 
			dropOutMask.matrix[0][i] = swap; 
		}
    }

	@Override
	public void setBatchSize(int size) {
		// TODO Auto-generated method stub
		
	}
}


class VectorizationLayer implements ANNLayer {
	private boolean set = false; 
	private int height, width; 
	private int numImgs; 
	
	public VectorizationLayer() {
		
	}
	
	@Override
	public DataContainer forwardProp(DataContainer forwardPropData, boolean training) {
		// TODO Auto-generated method stub
		
		List<Mat> input = forwardPropData.getImgs(); 
		
		if (input.size() == 0) {
			throw new IllegalArgumentException("Passing empty input data"); 
		}
		
		if (!set) {
			Mat matrix = input.get(0); 
			height = matrix.size[0]; 
			width = matrix.size[1]; 
			numImgs = input.size(); 
			set = true; 
		}
		
		Mat output = new Mat(1, height * width * numImgs, 0); 
		int idx = 0; 
		for (int i = 0; i < input.size(); i++) {
			Mat inMatrix = input.get(i); 
			for (int a = 0; a < inMatrix.size[0]; a++) {
				for (int b = 0; b < inMatrix.size[1]; b++) {
					output.matrix[0][idx] = inMatrix.matrix[a][b]; 
					idx++; 
				}
			}
		}
		
		return new DataContainer(output); 
	}

	@Override
	public DataContainer backProp(DataContainer backPropError, double learningRate) {
		// TODO Auto-generated method stub
		
		Mat vector = backPropError.getVec(); 
		if(vector.size[0] != 1 || vector.size[1] != height * width * numImgs) {
			throw new IllegalArgumentException("Vector length mismatch"); 
		}
		//System.out.println(vector);
		List<Mat> output = new ArrayList<Mat>(); 
		int idx = 0; 
		for (int i = 0; i < numImgs; i++) {
			Mat img = new Mat(height, width, 0); 
			for (int a = 0; a < height; a++) {
				for (int b = 0; b < width; b++) {
					img.matrix[a][b] = vector.matrix[0][idx]; 
					idx++; 
				}
			}
			output.add(img); 
		}
		
		return new DataContainer(output); 
	}

	@Override
	public ANNLayer copy() {
		// TODO Auto-generated method stub
		return this; 
	}

	@Override
	public void setBatchSize(int size) {
		// TODO Auto-generated method stub
		
	}
	
	
}
