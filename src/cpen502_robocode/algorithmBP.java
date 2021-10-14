// algorithmBP.java
// Author: Tom Sung

package cpen502_robocode;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class algorithmBP {
    // Variable Set-up: Representation, Learning rate, Momentum, and Iterations
    private String representation = "bipolar";
    private int iterations;
    private double momentum;
    private double learnRate;

    /** LUT variables and number of states defined here */
    final int x_state = 8;
    final int y_state = 6;
    final int dist_state = 4;
    final int bearing_state = 4;
    final int actions = 4;
    private String[] statesLUT;

    // Variable Set-up: Expected input/output; Arrays
    double[][] binary_input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double[][] binary_input_bias = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    double[] binary_output = {0.0, 1.0, 1.0, 0.0};
    double[][] bipolar_input = {{-1, -1}, {-1, +1}, {+1, -1}, {+1, +1}};
    double[][] bipolar_input_bias = {{-1, -1, 1}, {-1, +1, 1}, {+1, -1, 1}, {+1, +1, 1}};
    double[] bipolar_output = {-1, +1, +1, -1};
    double[][] input_expected;
    double[] output_expected;// = binary_output.clone();
    static String[][] LUT;

    boolean biasWeight = true; // Hard-coded; don't change

    public void setIterations(int iterations){
        this.iterations = iterations;
    }
    public void setMomentum(double momentum){
        this.momentum = momentum;
    }
    public void setLearnRate(double learnRate){
        this.learnRate = learnRate;
    }
    public void setRepresentation(boolean represent) {
        if (!represent) {
            this.representation = "binary";
        } else {
            this.representation = "bipolar";
        }

        if (representation.equals("bipolar")) {
            this.output_expected = this.bipolar_output.clone();
            if (biasWeight == false) {
                this.input_expected = this.bipolar_input.clone();
            } else {
                this.input_expected = this.bipolar_input_bias.clone();
            }
        } else {
            this.output_expected = this.binary_output.clone();
            if (biasWeight == false) {
                this.input_expected = this.binary_input.clone();
            } else {
                this.input_expected = this.binary_input_bias.clone();
            }
        }
    }
    public void setBiasWeight(boolean biasWeight){
        this.biasWeight = biasWeight;
    }

    public void readLUT(String filename){
        List<String[]> dataList = new ArrayList<String[]>();
        try{
            FileReader f = new FileReader(filename);
            BufferedReader readFile = new BufferedReader(f);
//            BufferedReader readFile = new BufferedReader(new FileReader(getDataFile(filename)));
            Scanner reader = new Scanner(readFile);
            while(reader.hasNextLine()){
                String data = reader.nextLine();
                String[] parse = data.split("   ");
//                String[] parse = data.split("");
                dataList.add(parse);
            }
            System.out.println("FILE READ OK: File read successful!");
        } catch (Exception e){
            System.out.println("FILE READ ERROR: File read error happened");
            e.printStackTrace();
        }

        String[][] data_output = new String[dataList.size()][2];
        for(int i = 0; i<dataList.size();i++){
            String[] parse;
            parse = dataList.get(i);
            data_output[i][0] = parse[0];
            data_output[i][1] = parse[1];
        }
        LUT = data_output;
    }

    public void LUTTrainingSet() {
//        this.LUT = new String[x_state*y_state*dist_state*bearing_state*actions][2]; //LUT array created
        //States array used to initialize states:

//        this.statesLUT = new String[x_state * y_state * dist_state * bearing_state * actions];
        this.input_expected = new double [LUT.length][6];
        this.output_expected = new double [LUT.length];

//        for (int i = 0; i<LUT.length;i++){
//            input_expected[i][1] = 0;
//        }

//        System.out.println("LUT created.");
//        System.out.println("This is LUT length: " + LUT.length);

        // Initialize values to 0
//        for (int i = 0; i < LUT.length; i++) LUT[i][1] = Double.toString(0.0);
        // Initialize states
//        int counter = 0;
        for (int i = 1; i <= x_state; i++) {
            for (int j = 1; j <= y_state; j++) {
                for (int k = 1; k <= dist_state; k++) {
                    for (int l = 1; l <= bearing_state; l++) {
                        for (int m = 1; m <= actions; m++) {
                            input_expected[i][0] = i/100;
                            input_expected[i][1] = j/100;
                            input_expected[i][2] = k/100;
                            input_expected[i][3] = l/90;
                            input_expected[i][4] = m/4;
                            input_expected[i][5] = 1;
//                            statesLUT[counter] = Integer.toString(i) + Integer.toString(j) + Integer.toString(k) +
//                                    Integer.toString(l) + Integer.toString(m);
//                            counter++;
                        }
                    }
                }
            }
        }
        for (int i = 1; i<LUT.length; i++) output_expected[i] = Double.parseDouble(String.valueOf(LUT[i][1]));
    }

    public void algorithm()
    {
        // Call upon class BackPropagation for the main algorithm and set up private variables in the class
        BackPropagation bp = new BackPropagation(5,14,this.learnRate,this.momentum,
                this.representation,this.biasWeight);
//        bp.setLearningRate(this.learnRate);
//        bp.setMomentum(this.momentum);
//        bp.setRepresentation(this.representation);
//        bp.setNumWeights(8, 4); // Hard-coded. 2 inputs * 4 hidden = 8 links
//        bp.setBiasWeight(this.biasWeight); // Do we want +1 bias weight?
        bp.setArrays();


        int[] epoch_num = new int[iterations];

        for (int h = 0; h < iterations; h++) {
            // Variable Set-up: Error
            double[] error_squared = new double[input_expected.length];
            double error = 1000;
            ArrayList<Double> error_all = new ArrayList<Double>();
            double desired_error = 0.05; //desired_error = 0.001;
            int epoch = 1;

            // Initializing weights
            bp.zeroWeights();
            bp.initializeWeights();

/*
        // Custom hard-coded weights in the beginning to verify work
        double[] inputWeight = {-0.33 ,0.18,
                0.42, -0.18,
                0.0, 0.02,
                -0.08, 0.39};
        double[] hiddenWeight = {0.25, -0.5, 0.41, -0.17};
        // set the weights in the private variables in BackPropagation.java
        for(int i = 0;i<inputWeight.length;i++) bp.setInputWeights(inputWeight[i],i);
        for(int i = 0;i<hiddenWeight.length;i++) bp.setHiddenWeights(hiddenWeight[i],i);


 */

            bp.savePrevWeights(); // Save previous weights for momentum calculations

            while (error > desired_error && epoch < 15000 + 1) {
                for (int i = 0; i < input_expected.length; i++) {
                    // Forward Propagation
                    bp.hiddenNeurons(input_expected[i]);
                    bp.outputNeuron();

                    double actual_output = bp.getFinalOutput();
//                System.out.println("The final output for pattern " + (i + 1) + " is: " + actual_output);

                    error_squared[i] = Math.pow((actual_output - output_expected[i]), 2);
//                System.out.println("Squared error for pattern " + (i + 1) + " is: " + error_squared[i]);

                    // Back Propagation
                    bp.sigmoidDerivative();
                    bp.deltaFinal(output_expected[i]);
                    bp.updateHiddenWeights();
                    bp.deltaHidden();
                    bp.updateInputWeights(input_expected[i]);
                    bp.savePrevWeights();
                    bp.saveNextWeights();
                    // You don't update weights back to +1... I was mistaken. You don't even need this function
//                    bp.updateBiasWeights(); // This will cause system not converge properly sometimes

                    // For debugging purposes
//                double finalOutput = bp.getFinalOutput();
//                System.out.println("For epoch " + epoch + " and pattern " + (i+1)
//                        + ", actual output: " + finalOutput + ". Expected output: " + output_expected[i]);
                }

                // Calculate Error
                error = bp.error(error_squared);
//            System.out.println("Epoch " + epoch + " error: " + error);
                epoch++;
                error_all.add(error);
            }

            epoch--;
            epoch_num[h] = epoch;

            // Getting total error and saving them into a separate file for easy graphing
            int j = error_all.size();
            double[] error_data = new double[j];
            for (int i = 0; i < error_all.size(); i++) error_data[i] = error_all.get(i);
            bp.fileWrite(error_data, "error.txt");
            System.out.println("The system reached an error of " + error + " in " + epoch + " epochs.");

        }
//        BackPropagation bp = new BackPropagation();
        bp.fileWrite(epoch_num, "epochIterations.txt");
        double epoch_sum = 0;
        for(int i = 0;i<epoch_num.length;i++){
            epoch_sum += epoch_num[i];
        }
        System.out.println("Out of " + iterations + " iterations, the average epoch is: " + epoch_sum/iterations);

    }
}


