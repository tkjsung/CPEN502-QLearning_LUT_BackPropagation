package cpen502_robocode;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

/**
 * BackPropagation.java - Back Propagation Algorithm
 * @author Tom (Ke-Jun) Sung
 * @since 2020-12-08
 */
public class BackPropagation{
    private double learn_rate, momentum;
    private String representation = new String();
    private boolean biasWeight;

    private int numInput = 2; // Default input/hidden amount of neurons
    private int numHidden = 4;
    private int numHiddenWeights = 4; // Default input/hidden weight values
    private int numInputWeights = 8;

    private double [] hiddenWeights;
    private double [] inputWeights;
    private double [] hiddenWeights_prev;
    private double [] inputWeights_prev;
    private double [] hiddenWeights_next;
    private double [] inputWeights_next;
    private double [] hiddenOutputs;
    private double finalOutput, finalDelta, finalDerivative;
    private double [] hiddenDelta;
    private double [] hiddenDerivative;

    // Get the right representation (binary or bipolar)
    public BackPropagation(int NumInputs, int NumHidden, double LearningRate, double Momentum,
                           String representation, boolean biasTerm){

        this.numInputWeights = NumInputs;
        this.numHiddenWeights = NumHidden;
        this.learn_rate = LearningRate;
        this.momentum = Momentum;
        this.representation = representation;
        this.biasWeight = biasTerm;
        setNumWeights(NumInputs, NumHidden);

    }

    public void setNumWeights(int input, int hidden){
        this.numInput = input;
        this.numHidden = hidden;
        this.numInputWeights = input*hidden;
        this.numHiddenWeights = hidden;
        if(this.biasWeight){
            this.numInputWeights = (input+1)*hidden;
            this.numHiddenWeights = hidden + 1;
        }
    }

    public void setArrays(){
        hiddenWeights = new double[this.numHiddenWeights];
        inputWeights = new double[this.numInputWeights];
        hiddenWeights_prev = new double[this.numHiddenWeights];
        inputWeights_prev = new double[this.numInputWeights];
        hiddenWeights_next = new double[this.numHiddenWeights];
        inputWeights_next = new double[this.numInputWeights];
        hiddenOutputs = new double[this.numHiddenWeights];
        hiddenDelta = new double[this.numHiddenWeights];
        hiddenDerivative = new double[this.numHiddenWeights];
    }

    // Forward Propagation
    // Step 1: Initialize weights
    public void initializeWeights(){
        Random rng = new Random();
        double lowerbound = -0.5;
        double upperbound = 0.5;
        double randomNum;

        for (int i = 0; i<this.inputWeights.length; i++){
            randomNum = lowerbound + (upperbound - lowerbound) * rng.nextDouble();
            this.inputWeights[i] = randomNum;
        }
        for(int i = 0; i<this.hiddenWeights.length; i++) {
            randomNum = lowerbound + (upperbound - lowerbound) * rng.nextDouble();
            this.hiddenWeights[i] = randomNum;
        }
    }

    // Aside: Retrieve or write to weight array/variable
    public double[] getHiddenWeights(){
        return this.hiddenWeights;
    }
    public double[] getInputWeights(){
        return this.inputWeights;
    }
    public void setHiddenWeights(double weight, int index){
        this.hiddenWeights[index] = weight;
    }
    public void setInputWeights(double weight, int index){
        this.inputWeights[index] = weight;
    }
    // We need to set weights to 0 in some situations because the arrays use the same memory location
    public void zeroWeights(){
        for(int i = 0; i<this.hiddenWeights.length; i++) this.hiddenWeights[i] = 0;
        for(int i = 0; i<this.inputWeights.length; i++) this.inputWeights[i] = 0;
    }


    // Step 2: Calculate S_j, the "x" in sigmoid function, and the entire sigmoid function
    // Each call to calculateOutput() finds the output for each neuron
    // This outputs calculated sigmoid function
    public double calculateOutput(double[] input, double[] weight){
        double summation = 0;
        if(input.length != weight.length){
            throw new ArrayIndexOutOfBoundsException("The two arrays do not have the same number of indices");
        }
        else {
            for (int i = 0; i < input.length; i++) summation = summation + (input[i]*weight[i]);
            double result = sigmoid(summation);
            return result;
        }
    }

    // Sigmoid function, which includes both binary and bipolar representations
    public double sigmoid(double x) {
        double y = 0;

        if (this.representation.equals("binary") == true) {
            y = 1 / (1 + Math.exp(-x)); // the function itself...for binary representation
        } else if (this.representation.equals("bipolar") == true) {
            y = (1-Math.exp(-x))/(1+(Math.exp(-x))); // bipolar representation
        }

        return y;
    }

    // Aside: Retrieve or write hidden/final outputs
    public void setHiddenOutputs(double x, int index){
        this.hiddenOutputs[index] = x;
    }
    public double[] getHiddenOutputs(){
        return this.hiddenOutputs;
    }
    public void setFinalOutput(double x){
        this.finalOutput = x;
    }
    public double getFinalOutput(){
        return this.finalOutput;
    }

    // Takes the input, and performs sigmoid function calculations for all hidden neurons
    // Then the output of the sigmoid function is stored in the hiddenOutput array
    public void hiddenNeurons(double[] x_input){
        double[] weights = new double[x_input.length];
        for(int j = 0; j<numHidden; j++) {
            if(this.biasWeight==false) {
                for(int k = 0; k<numInput; k++) weights[k] = this.inputWeights[(numInput * j) + k];
                    double result = calculateOutput(x_input, weights);
                    setHiddenOutputs(result, j);

            } else {
                for(int k = 0; k<(numInput+1); k++) weights[k] = this.inputWeights[((numInput+1) * j) + k];
                double result = calculateOutput(x_input, weights);
                setHiddenOutputs(result, j);

            }
        }
    }

    // The output value for the output neuron
    public void outputNeuron(){
        double result = calculateOutput(this.hiddenOutputs, this.hiddenWeights);
        setFinalOutput(result);
    }


    // Step 3: Calculate error of the forward propagation process
    public double error(double[] error_squared){
        double result = 0;
        for(int i = 0; i<error_squared.length; i++){
            result = result + error_squared[i];
        }
        result = result*0.5;
        return result;
    }

    // Back Propagation
    // Step 1: Calculate all f'(x)
    public void sigmoidDerivative(){
        if(this.representation.equals("binary")) {
            for (int i = 0; i < this.hiddenOutputs.length; i++)
                this.hiddenDerivative[i] = (this.hiddenOutputs[i]) * (1 - (this.hiddenOutputs[i]));
            this.finalDerivative = (this.finalOutput) * (1 - (this.finalOutput));
        } else if(this.representation.equals("bipolar")){
            for (int i = 0; i<this.hiddenOutputs.length;i++)
                this.hiddenDerivative[i] = 0.5*(1+this.hiddenOutputs[i])*(1-this.hiddenOutputs[i]);
            this.finalDerivative = 0.5*(1+this.finalOutput)*(1-this.finalOutput);
        }
    }

    // Step 2: Calculate delta for final output
    public void deltaFinal(double output_expected){
        this.finalDelta = (output_expected-this.finalOutput)*this.finalDerivative;
    }

    // Step 3: Update hidden weights given final output
    // TODO: Make sure the bias weight is +1.
    public void updateHiddenWeights(){
        for (int i = 0; i<this.hiddenWeights.length;i++)
            this.hiddenWeights_next[i] = this.hiddenWeights[i] +
                    this.learn_rate*this.finalDelta*this.hiddenOutputs[i] +
                    (this.momentum*(this.hiddenWeights[i]-this.hiddenWeights_prev[i]));
    }

    // Step 4: Calculate delta for hidden outputs
    // Only this needs to be changed if I'm switching over to the other algorithm
    public void deltaHidden(){
        for(int i = 0;i<this.hiddenWeights.length;i++) {
            this.hiddenDelta[i] = this.finalDelta * this.hiddenWeights_next[i] * this.hiddenDerivative[i];
        }
    }

    // Step 5: Update input weights
    public void updateInputWeights(double[] x_input){
        int k = 0;
        if (!biasWeight) {
            for (int i = 0; i < numHidden; i++) {
                for (int index = 0; index < numInput; index++) {
                    this.inputWeights_next[k + index] = this.inputWeights[k + index] +
                            (this.learn_rate * this.hiddenDelta[i] * x_input[index]) +
                            (this.momentum * (this.inputWeights[k + index] - this.inputWeights_prev[k + index]));
                }
                k = k + numInput;

            }
        } else {
            for (int i = 0; i < numHidden; i++) {
                for (int index = 0; index < numInput+1; index++) {
                    this.inputWeights_next[k + index] = this.inputWeights[k + index] +
                            (this.learn_rate * this.hiddenDelta[i] * x_input[index]) +
                            (this.momentum * (this.inputWeights[k + index] - this.inputWeights_prev[k + index]));
                }
                k = k + (numInput + 1);

            }
        }
    }

    // We need to keep a record of previous weights (place after momentum calc)
    public void savePrevWeights(){
        this.hiddenWeights_prev = this.hiddenWeights.clone();
        this.inputWeights_prev = this.inputWeights.clone();
    }

    public void saveNextWeights(){
        this.hiddenWeights = this.hiddenWeights_next.clone();
        this.inputWeights = this.inputWeights_next.clone();
    }

    // Creating, reading, and saving files
    public void fileCreate(String filename){
        try {
            File testFile = new File(filename);
            if (testFile.createNewFile()) {
                System.out.println("FILE: File created: " + testFile.getName());
            } else {
                System.out.println("FILE: File already exists");
            }
            System.out.println("FILE: File path: " + testFile.getAbsolutePath());
        } catch (IOException e) {
            System.out.println("FILE: An error occurred.");
            e.printStackTrace();
        }
    }

    public void fileWrite(double[] data, String filename){
        String[] data_String = new String[data.length];
        String convert = new String();

        for(int i = 0; i<data.length;i++) data_String[i] = convert.valueOf(data[i]);

        try{
            FileWriter writeFile = new FileWriter(filename);
            for(int i = 0;i<data_String.length;i++){
                writeFile.write(data_String[i] + "\n");
            }
            writeFile.close();
        } catch (IOException e){
            System.out.println("FILE: Error in writing file");
            e.printStackTrace();
        }
    }

    public void fileWrite(int[] data,String filename){
        String[] data_String = new String[data.length];
        String convert = new String();

        for(int i = 0; i<data.length;i++) data_String[i] = convert.valueOf(data[i]);

        try{
            FileWriter writeFile = new FileWriter(filename);
            for(int i = 0;i<data_String.length;i++){
                writeFile.write(data_String[i] + "\n");
            }
            writeFile.close();
        } catch (IOException e){
            System.out.println("FILE: Error in writing file");
            e.printStackTrace();
        }
    }

    public void fileDelete(String filename){
        File deleteFile = new File(filename);
        if (deleteFile.delete()){
            System.out.println("Deleted the file: " + deleteFile.getName());
        } else {
            System.out.println("Failed to delete the file.");
        }
    }

    public double[] fileRead(String filename){
        ArrayList<Double> data_arraylist = new ArrayList<Double>();
        double parse;
        int j = 0;

        try{
            File readFile = new File(filename);
            Scanner reader = new Scanner(readFile);
            while(reader.hasNextLine()){
                String data_String = reader.nextLine();
                parse = Double.parseDouble(data_String);
                data_arraylist.add(parse);
            }

        } catch (FileNotFoundException e){
            System.out.println("FILE: File read error happened");
            e.printStackTrace();
        }

        j = data_arraylist.size();
        double [] data = new double[j];

        for(int i = 0;i<data_arraylist.size();i++) data[i] = data_arraylist.get(i);
        return data;
    }

}
