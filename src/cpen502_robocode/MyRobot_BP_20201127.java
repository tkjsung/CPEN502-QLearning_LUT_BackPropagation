package cpen502_robocode;

import robocode.*;
import robocode.RobocodeFileWriter;

import java.awt.geom.Point2D;
import java.io.*;
import java.util.*;

/**
 * MyRobot_BP_20201127.java - Reinforcement Learning with Back Propagation
 * @author Tom (Ke-Jun) Sung
 * @since 2020-12-08
 */
public class MyRobot_BP_20201127 extends AdvancedRobot{
    /** LUT variables and number of states defined here */
    final int x_state = 8; // unused
    final int y_state = 6; // unused
    final int dist_state = 4; // unused
    final int energy_state = 4; // unused
    final int bearing_state = 4; // unused
    final int actions = 4;
    private static String[][] LUT;
    final int numInputs = 5; // includes state-action. Bias weight not counted

    /** Quantized values for LUT. Default values are given to avoid errors/warnings */
    private int quantizedX = 1;
    private int quantizedY = 1;
    private int quantizedDistance = 1;
    private int quantizedBearing = 4;

    /** Status of my robot */
    private double robotBearing = 0.0; // Get bearing. Initialized to 0.
    private double robotAbsoluteBearing = 0.0; // Get absolute bearing. Initialized to 0.
    private double robotHeading = 0.0; // Get heading. Initialized to 0.
    private double robotXPos = 0.0; // Get x-coordinate for current location. Initialized to 0.
    private double robotYPos = 0.0; // Get y-coordinate for current location. Initialized to 0.
    private double robotDistToEnemy = 0.0; // Get distance to enemy from current location. Initialized to 0.

    /** Current/Next state action values */
    private int currentState = 11141;
    private int nextState = 11141;
    private double currentStateValue = 0.0;
    private double nextStateValue = 0.0;

    /** Neural Network Required Matrices */
    private static double [][] nn_input;
    private static double [] nn_output; // Expected value.
    private static double [][] nn_input_next;
    private static double [] nn_output_next;
    private double [] nn_actualOutput = new double[actions];
    private int nn_index;
    private static double [][] nn_history_input; // First [] should be "n", second one is [6], 4s, 1a, 1 bias
    private static double [] nn_history_output;
    private static double [] nn_history_max;
    private static int historyInit = 0;

    /** Other required variables */
    String[][] selectedStates = new String[this.actions][2]; // Second column: "expected output" used for BP training
    private static int initial = 0;
    private int hasRoundEnded = 0;
    private double rewards = 0;
    private double totalReward = 0;
    private ArrayList<String> winRecord = new ArrayList<String>();
    private ArrayList<String> rewardRecord = new ArrayList<String>();
    private static double errorRecord = 100;

    /** Variables that can be changed to exhibit different behaviour */
    final double stepSize = 0.15; // Q-Value Step Size
    final double gamma = 0.9; // Discount factor. Determines if future rewards is worth it or not.
    final double explore = 0.0; // Explore Probability
    final boolean offPolicy = true; // Off-/On-Policy update (TD)
    final double learnRate = 0.00005; // BP variable. learnRate = 0.000001 for Corners. 0.00001 for Tracker
    final double momentum = 0.9; // BP variable
    final int n = 10; // Will be used later for Q5(e), saving history of BP

    BackPropagation bp = new BackPropagation(this.numInputs,25,this.learnRate,this.momentum,
            "bipolar",true);

    public void run() {
        if(initial == 0) {
            nn_input = new double[this.actions][6]; // 6 due to 4 states, 1 action, 1 bias weight
            nn_output = new double[this.actions];
            nn_input_next = nn_input.clone();
            nn_output_next = nn_output.clone();
            nn_history_input = new double[n][6];
            nn_history_output = new double[n];
            nn_history_max = new double[n];
            readLUT("LUT.txt"); // Should be only read once
            initial = 1;
        }
        bp.setArrays();

        /* The following few functions are used for the beginning of training.
         * Please uncomment to get a fresh new file.
         * Best way: Toggle a breakpoint on readHiddenWeights(). Enter Debug mode.
         * Add only this tank into Robocode battle to save time. */
//        bp.zeroWeights(); bp.initializeWeights();
//        writeHiddenWeights("hiddenWeights.txt"); writeInputWeights("inputWeights.txt");
//        initializeRecordFile("winCounter.txt"); initializeRecordFile("rewardTracker.txt");
//        initializeRecordFile("errorRecord.txt");


        // Initializing state. (more like predicting initial state)
        readHiddenWeights("hiddenWeights.txt");
        readInputWeights("inputWeights.txt");
        readWinRecord("winCounter.txt");
        readRewardRecord("rewardTracker.txt");
//        readErrorRecord("errorRecord.txt");
        updateRobotStatus();
        turnGunRight(360);// Initialize the states by scanning the environment
        quantize();
        getSelectedStatesAction(); // Add actions to states and get expected values. nn_output initialized here.
        convertStates2NN(); // Function here to put some input into NN

        for(int i = 0; i< nn_input_next.length;i++) {
            bp.hiddenNeurons(nn_input_next[i]);
            bp.outputNeuron();
            this.nn_actualOutput[i] = bp.getFinalOutput();
        }

        int maxState = 11141; double maxStateValue = -1000; // Initialized to avoid warnings

        // Take maximum of current state (epsilon-greedy).
        for(int i = 0; i<actions; i++){
            if(this.nn_actualOutput[i] > maxStateValue){
                maxState = Integer.parseInt(this.selectedStates[i][0]);
//                maxStateValue = Double.parseDouble(this.selectedStates[i][1]);
                maxStateValue = this.nn_actualOutput[i];
                this.nn_index = i;
            }
        }

        // We finally now have the state-action pair, given the selected policy.
        this.currentState = maxState;
        this.currentStateValue = maxStateValue;
        bp.savePrevWeights();
        nn_input = nn_input_next.clone();
        nn_output = nn_output_next.clone();
        int temp = 0; double rmsError = 0;

        while (true) {
            this.rewards = 0;
            System.out.println("REWARDS RESET: Rewards reset to 0 at beginning of turn.");
            // Get the probability to see if we explore.
            double exploreChance = randomNumber(0,1);
            // This means we don't explore and we take maximum of the states
            if(exploreChance > this.explore){
                System.out.println("NO EXPLORATION");

                int actionDigit = getActionDigit(this.currentState); // Get the action digit from the state
                robotAction(actionDigit); // Perform action

                // Observe the environment and the rewards
                turnGunRight(360);
                updateRobotStatus();
                quantize();
                getSelectedStatesAction();
                convertStates2NN();

                // BP Forward Propagation to get summed output from sigmoid function
                for(int i = 0; i < nn_input_next.length;i++) {
                    bp.hiddenNeurons(nn_input_next[i]);
                    bp.outputNeuron();
                    this.nn_actualOutput[i] = bp.getFinalOutput();
                }

                // System out for checking purposes
                for(int i = 0; i<nn_actualOutput.length; i++){
                    System.out.println(nn_actualOutput[i]);
                }

                maxState = 11141; maxStateValue = -1000; // Initialized to avoid warnings

                // Take maximum of current state (epsilon-greedy).
                temp = 0;
                for(int i = 0; i < actions; i++){
                    if(this.nn_actualOutput[i] > maxStateValue){
                        maxState = Integer.parseInt(this.selectedStates[i][0]);
                        maxStateValue = nn_actualOutput[i];
                        temp = i;
                    }
                }
//                System.out.println("Maximum - State-action: " + maxState + " , Value: " + maxStateValue);
//                System.out.println("nn_input_next: " + Arrays.toString(nn_input_next[temp]));
                this.nextState = maxState;
                this.nextStateValue = maxStateValue;

                // Now update states (Q equation).
                System.out.println("UPDATING Q-EQUATION");
                double tempStateValue = this.currentStateValue + this.stepSize*(this.rewards +
                        this.gamma*this.nextStateValue-this.currentStateValue);

                System.out.println("New Q value: " + tempStateValue + " LUT value: " + nn_output[nn_index]);

                // Get RMS Error
                rmsError = tempStateValue - nn_output[nn_index];
                if(Math.abs(rmsError) < errorRecord){
                    errorRecord = Math.abs(rmsError);
                }

                // Backpropagation History
                if(historyInit == 0){
                    for(int i = 0; i<this.n; i++){
                        nn_history_input[i] = nn_input[nn_index]; // nn_index for index only for this case
                        nn_history_output[i] = nn_output[nn_index]; // nn_index for index ony for this case
                        nn_history_max[i] = tempStateValue;
                    }
                    historyInit = 1;
                }
                else {
                    for(int i = 0; i<this.n-1; i++){
                        nn_history_input[i] = nn_history_input[i+1];
                        nn_history_output[i] = nn_history_output[i+1];
                        nn_history_max[i] = nn_history_max[i+1];
                    }
                    nn_history_input[n-1] = nn_input[nn_index];
                    nn_history_output[n-1] = nn_output[nn_index];
                    nn_history_max[n-1] = tempStateValue;
                }

                for(int i = 0; i < this.n; i++) {
//                    bp.setFinalOutput(tempStateValue);
                    bp.setFinalOutput(nn_history_max[i]);
                    // BackPropagation here.
                    bp.sigmoidDerivative();
//                    bp.deltaFinal(nn_output[nn_index]);
                    bp.deltaFinal(nn_history_output[i]);
                    bp.updateHiddenWeights();
                    bp.deltaHidden();
//                    bp.updateInputWeights(nn_input[nn_index]);
                    bp.updateInputWeights(nn_history_input[i]);
                    bp.savePrevWeights();
                    bp.saveNextWeights();
                }
            }
            // Exploration.
            else {
                System.out.println("EXPLORATION STEP: Exploration step is done here.");
                // Steps:
                // 1. Generate random number for a random action
                // 2. Get the state and its value for the random action
                // 3. Also get the maximum state/value for off-policy
                // 4. If statement. If on-policy, update using the random action's value
                //    Else, off-policy, update using maximum value to the previous state.
                int exploreState = this.currentState; // Original path for non-explore (not used)
                double exploreStateValue = this.currentStateValue; // "max" value for off-policy

                // Get random number to explore actions with
                int actionDigit = randomNumberAction(this.actions);
                // Put the random action in the current state by replacing the last digit
                int stateValue = this.currentState;
                String tempString = String.valueOf(stateValue);
                String[] tempStringArray = tempString.split("");
                tempStringArray[4]= String.valueOf(actionDigit); // TODO: replace 4 with last element of state-action
                tempString = String.join("",tempStringArray);
                // Update the this.currentState and this.currentStateValue (otherwise it would take max)
                this.currentState = Integer.parseInt(tempString);
                for(int i = 0; i<LUT.length; i++){
                    if(Integer.parseInt(LUT[i][0]) == this.currentState){
                        this.currentStateValue = Double.parseDouble(LUT[i][1]);
                    }
                }
                this.nn_index = actionDigit - 1;
                robotAction(actionDigit); // Perform action

                // Observe the environment and the rewards (Scan environment for next state)
                turnGunRight(360);
                updateRobotStatus();
                quantize();
                getSelectedStatesAction();
                convertStates2NN();

                // BP Forward Propagation to get summed output from sigmoid function
                for(int i = 0; i< nn_input_next.length;i++) {
                    bp.hiddenNeurons(nn_input_next[i]);
                    bp.outputNeuron();
                    this.nn_actualOutput[i] = bp.getFinalOutput();
                }

                maxState = 11141; maxStateValue = -1000; // Initialized to avoid warnings
                // Get the maximum state-action pair (to ensure proper transition into non-explore state)
                temp = 0;
                for(int i = 0; i<actions; i++){
                    if(this.nn_actualOutput[i] > maxStateValue){
                        maxState = Integer.parseInt(this.selectedStates[i][0]);
                        maxStateValue = nn_actualOutput[i];
                        temp = i;
                    }
                }
//                System.out.println("Explore state-action: "+this.currentState + ", Value: " + this.currentStateValue);
//                System.out.println("nn_input_next: " + Arrays.toString(nn_input_next[nn_index]));

                // Explore state: next action's maximum values
                this.nextState = maxState;
                this.nextStateValue = maxStateValue;

                // Now update states (Q equation)
                double tempStateValue = 0;
                if(offPolicy){
                    System.out.println("Off policy Q-value update.");
                    tempStateValue = this.currentStateValue + this.stepSize*(this.rewards +
                            this.gamma*exploreStateValue-this.currentStateValue);
//                    System.out.println(exploreStateValue + " vs. " + this.currentStateValue);
                }
                else {
                    System.out.println("On policy Q-value update.");
                    tempStateValue = this.currentStateValue + this.stepSize*(this.rewards +
                            this.gamma*this.nextStateValue-this.currentStateValue);
                }

                rmsError = tempStateValue - nn_output[nn_index];
                if(Math.abs(rmsError) < errorRecord){
                    errorRecord = Math.abs(rmsError);
                }

                // Backpropagation History
                if(historyInit == 0){
                    for(int i = 0; i<this.n; i++){
                        nn_history_input[i] = nn_input[nn_index]; // nn_index for index only for this case
                        nn_history_output[i] = nn_output[nn_index]; // nn_index for index ony for this case
                        nn_history_max[i] = tempStateValue;
                    }
                    historyInit = 1;
                }
                else {
                    for(int i = 0; i<this.n-1; i++){
                        nn_history_input[i] = nn_history_input[i+1];
                        nn_history_output[i] = nn_history_output[i+1];
                        nn_history_max[i] = nn_history_max[i+1];
                    }
                    nn_history_input[n-1] = nn_input[nn_index];
                    nn_history_output[n-1] = nn_output[nn_index];
                    nn_history_max[n-1] = tempStateValue;
                }

                for(int i = 0; i < this.n; i++) {
//                    bp.setFinalOutput(tempStateValue);
                    bp.setFinalOutput(nn_history_max[i]);
                    // BackPropagation here.
                    bp.sigmoidDerivative();
//                    bp.deltaFinal(nn_output[nn_index]);
                    bp.deltaFinal(nn_history_output[i]);
                    bp.updateHiddenWeights();
                    bp.deltaHidden();
//                    bp.updateInputWeights(nn_input[nn_index]);
                    bp.updateInputWeights(nn_history_input[i]);
                    bp.savePrevWeights();
                    bp.saveNextWeights();
                }
            } // end else loop for exploration

            // Save rewards. Save wins. Save error.
            // Assign next state to current state
            this.currentState = this.nextState;
            this.currentStateValue = this.nextStateValue;
            this.totalReward += this.rewards;
            nn_input = nn_input_next.clone();
            nn_output = nn_output_next.clone();
            this.nn_index = temp;
            writeHiddenWeights("hiddenWeights.txt");
            writeInputWeights("inputWeights.txt");
            writeErrorRecord("errorRecord.txt",rmsError);
        } // end while()
    } // end run()

    /** convertStates2NN: Get the inputs into the Neural Network */
    public void convertStates2NN(){
        for(int i = 0; i<this.actions; i++) {
            nn_input_next[i][0] = (double) this.quantizedX /100;
            nn_input_next[i][1] = (double) this.quantizedY /100;
            nn_input_next[i][2] = (double) this.quantizedDistance /100;
            nn_input_next[i][3] = (double) this.quantizedBearing /90;
            nn_input_next[i][4] = (double) (i + 1) /4;
            nn_input_next[i][5] = 1.0;
        }
    }

    /**
     * getSelectedStates: Given the quantized states, attach possible actions to them. Then put their values in a
     * matrix selectedStates[][]. This will help with either on- or off-policy situations. */
    public void getSelectedStatesAction(){
        for(int i = 0; i<this.actions; i++) {
            this.selectedStates[i][0] = Integer.toString(this.quantizedX) + Integer.toString(this.quantizedY) +
                    Integer.toString(this.quantizedDistance) + Integer.toString(this.quantizedBearing) +
                    Integer.toString(i+1);
        }
        // We're getting the expected value from the LUT and placing them in the second column
        // No need for any changes to the following
        for(int j = 0; j<this.actions; j++){
            for(int i = 0; i<LUT.length; i++){
                if(LUT[i][0].equals(this.selectedStates[j][0])){
                    this.selectedStates[j][1] = LUT[i][1];
                    nn_output_next[j] = Double.parseDouble(this.selectedStates[j][1]);
                }
            }
        }
    }

    /** getActionDigit: Get the last digit of the state value in order to perform action */
    public int getActionDigit(int stateValue){
        String tempString = String.valueOf(stateValue);
        String[] tempStringArray = tempString.split("");
        int actionDigit = Integer.parseInt(tempStringArray[tempStringArray.length-1]);
        return actionDigit;
    }

    // Random Number Generation Functions
    /** randomNumber: Function with double type. Returns value given lower and upper bound
     * Mainly used for the probability of the explore step */
    public double randomNumber(double lowerbound, double upperbound){
        Random rng = new Random();
        double randomNum = lowerbound + (upperbound - lowerbound) * rng.nextDouble();
        return randomNum;
    }

    /** randomNumberAction(): Choose a random action to perform in the explore step */
    public int randomNumberAction(int upperbound){
        Random rng = new Random();
        int randomNum = rng.nextInt(upperbound) + 1;
        return randomNum;
    }


    /** Action States
     * 1. Move towards the robot
     * 2. Move away from the robot
     * 3. 90 degrees from the robot, move forward
     * 4. 90 degrees from the robot, move backward */
    public void robotAction(int actionState){
        double turn = 0;

        switch(actionState) {
            case 1:
                turnRight(this.robotBearing);
                shootEnemy();
                ahead(150);
                shootEnemy();
                break;
            case 2:
                turn = 180 - Math.abs(this.robotBearing);
                turnRight(this.robotBearing);
                shootEnemy();
                back(150);
                shootEnemy();
                break;
            case 3:
                turn = 90 - Math.abs(this.robotBearing);
                // If heading is 0 and bearing is 40, we turn 180-40=140 left
                // If heading is 0 and bearing is -40, we turn 180-abs(40)=140 right
                if (this.robotBearing >= 0) {
                    turnGunRight(90);
                    turnLeft(turn);
                    shootEnemy();
                    turnGunLeft(90);
                } else {
                    turnGunLeft(90);
                    turnRight(turn);
                    shootEnemy();
                    turnGunRight(90);
                }
                ahead(150);
                break;
            case 4:
                turn = 90 - Math.abs(this.robotBearing);
                // If heading is 0 and bearing is 40, we turn 180-40=140 left
                // If heading is 0 and bearing is -40, we turn 180-abs(40)=140 right
                if (this.robotBearing >= 0) {
                    turnGunRight(90);
                    turnLeft(turn);
                    shootEnemy();
                    turnGunLeft(90);
                } else {
                    turnGunLeft(90);
                    turnRight(turn);
                    shootEnemy();
                    turnGunRight(90);
                }
                back(150);
                break;
        }
    }

    /** shootEnemy: Shoot the enemy using the gun. Called whenever the gun needs to shoot */
    public void shootEnemy(){
        switch (this.quantizedDistance){
            case 1:
                fire(3);
                break;
            case 2:
                fire(2);
                break;
            case 3:
                fire(1);
                break;
            case 4:
                fire(0.1);
                fire(0.1);
                break;
            default:
                fire(1);
                break;
        }
    }

    /** updateRobotStatus: Update my robot's parameters. */
    public void updateRobotStatus(){
        this.robotHeading = getHeading();
        this.robotXPos = getX();
        this.robotYPos = getY();
    }

    /** onScannedRobot: What to do when you see another robot */
    public void onScannedRobot (ScannedRobotEvent e){
        this.robotBearing = e.getBearing(); // In angles
        this.robotDistToEnemy = e.getDistance();

        double angle = Math.toRadians((getHeading() + this.robotBearing % 360));
        double enemyX = (this.robotXPos + Math.sin(angle) * this.robotDistToEnemy);
        double enemyY = (this.robotYPos + Math.cos(angle) * this.robotDistToEnemy);
        this.robotAbsoluteBearing = absoluteBearing(getX(), getY(), enemyX, enemyY);

    }

    /** onHitByBullet: What to do when you're hit by a bullet */
    public void onHitByBullet (HitByBulletEvent e){
        System.out.println("GOT HIT: My robot got hit by the enemy.");
        this.rewards -= 1;
    }

    /** onHitRobot: What to do when you hit the enemy robot */
    public void onHitRobot (HitRobotEvent e){
        System.out.println("RAMMED ROBOT: My robot rammed the enemy.");
        this.rewards -= 0.25;
    }

    /** onBulletHit: What to do when the robot hits the enemy */
    public void onBulletHit (BulletHitEvent e){
        System.out.println("HIT ENEMY TANK: My robot got a hit on the enemy.");
        this.rewards += 1;
    }

    /** onHitHall: What to do when the robot hits the wall */
    public void onHitWall (HitWallEvent e){
        System.out.println("WALL: My robot hit a wall.");
        this.rewards -= 0.75;
        double x = this.getX();
        double y = this.getY();
        double width = this.getBattleFieldWidth();
        double height = this.getBattleFieldHeight();

        if(y < 90){
            turnLeft(getHeading() % 90);
            if(getHeading()==0) turnLeft(0);
            if(getHeading()==90) turnLeft(90);
            if(getHeading()==180) turnLeft(180);
            if(getHeading()==270) turnRight(90);
            ahead(150);

            if ((this.getHeading()<180)&&(this.getHeading()>90)) this.setTurnLeft(90);
            else if((this.getHeading()<270)&&(this.getHeading()>180)) this.setTurnRight(90);

        }
        else if(y > height-90){
            if((this.getHeading()<90)&&(this.getHeading()>0)) this.setTurnRight(90);
            else if((this.getHeading()<360)&&(this.getHeading()>270)) this.setTurnLeft(90);
            turnLeft(getHeading() % 90);
            if(getHeading()==0) turnRight(180);
            if(getHeading()==90) turnRight(90);
            if(getHeading()==180) turnLeft(0);
            if(getHeading()==270) turnLeft(90);
            ahead(150);
        }
        else if(x < 90){
            turnLeft(getHeading() % 90);
            if(getHeading()==0) turnRight(90);
            if(getHeading()==90) turnLeft(0);
            if(getHeading()==180) turnLeft(90);
            if(getHeading()==270) turnRight(180);
            ahead(150);
        }
        else if(x >width-90){
            turnLeft(getHeading() % 90);
            if(getHeading()==0) turnLeft(90);
            if(getHeading()==90) turnLeft(180);
            if(getHeading()==180) turnRight(90);
            if(getHeading()==270) turnRight(0);
            ahead(150);
        }

    }

    /** onRoundEnded: What to do when the battle round ends */
    public void onRoundEnded (RoundEndedEvent e){
        System.out.println("ROUND ENDED: The round has ended.");
    }

    /** onWin: What to do when the robot wins */
    public void onWin (WinEvent e){
        if(this.hasRoundEnded == 0) {
            System.out.println("VICTORY: My robot has won.");
            writeWinRecord("winCounter.txt", 1);
            this.hasRoundEnded = 1;

            // Terminal states update (use greedy update)
            this.rewards += 1;
            double tempStateValue = this.currentStateValue + this.stepSize*(this.rewards +
                    this.gamma*this.nextStateValue-this.currentStateValue);

            // Backpropagation History
            if(historyInit == 0){
                for(int i = 0; i<this.n; i++){
                    nn_history_input[i] = nn_input[nn_index]; // nn_index for index only for this case
                    nn_history_output[i] = nn_output[nn_index]; // nn_index for index ony for this case
                    nn_history_max[i] = tempStateValue;
                }
                historyInit = 1;
            }
            else {
                for(int i = 0; i<this.n-1; i++){
                    nn_history_input[i] = nn_history_input[i+1];
                    nn_history_output[i] = nn_history_output[i+1];
                    nn_history_max[i] = nn_history_max[i+1];
                }
                nn_history_input[n-1] = nn_input[nn_index];
                nn_history_output[n-1] = nn_output[nn_index];
                nn_history_max[n-1] = tempStateValue;
            }

            for(int i = 0; i < this.n; i++) {
//                    bp.setFinalOutput(tempStateValue);
                bp.setFinalOutput(nn_history_max[i]);
                // BackPropagation here.
                bp.sigmoidDerivative();
//                    bp.deltaFinal(nn_output[nn_index]);
                bp.deltaFinal(nn_history_output[i]);
                bp.updateHiddenWeights();
                bp.deltaHidden();
//                    bp.updateInputWeights(nn_input[nn_index]);
                bp.updateInputWeights(nn_history_input[i]);
                bp.savePrevWeights();
                bp.saveNextWeights();
            }
            writeHiddenWeights("hiddenWeights.txt");
            writeInputWeights("inputWeights.txt");

            // Track rewards
            this.totalReward += this.rewards;
            writeRewardRecord("rewardTracker.txt",this.totalReward);
        }
    }

    /** onDeath: What to do when the robot dies */
    public void onDeath (DeathEvent e){
        if(this.hasRoundEnded == 0) {
            System.out.println("ROBOT DEAD: My robot died.");
            writeWinRecord("winCounter.txt", 0);
            this.hasRoundEnded = 1;

            // Terminal states update (use greedy update)
            this.rewards -= 1;
            double tempStateValue = this.currentStateValue + this.stepSize*(this.rewards +
                    this.gamma*this.nextStateValue-this.currentStateValue);

            // Backpropagation History
            if(historyInit == 0){
                for(int i = 0; i<this.n; i++){
                    nn_history_input[i] = nn_input[nn_index]; // nn_index for index only for this case
                    nn_history_output[i] = nn_output[nn_index]; // nn_index for index ony for this case
                    nn_history_max[i] = tempStateValue;
                }
                historyInit = 1;
            }
            else {
                for(int i = 0; i<this.n-1; i++){
                    nn_history_input[i] = nn_history_input[i+1];
                    nn_history_output[i] = nn_history_output[i+1];
                    nn_history_max[i] = nn_history_max[i+1];
                }
                nn_history_input[n-1] = nn_input[nn_index];
                nn_history_output[n-1] = nn_output[nn_index];
                nn_history_max[n-1] = tempStateValue;
            }

            for(int i = 0; i < this.n; i++) {
//                    bp.setFinalOutput(tempStateValue);
                bp.setFinalOutput(nn_history_max[i]);
                // BackPropagation here.
                bp.sigmoidDerivative();
//                    bp.deltaFinal(nn_output[nn_index]);
                bp.deltaFinal(nn_history_output[i]);
                bp.updateHiddenWeights();
                bp.deltaHidden();
//                    bp.updateInputWeights(nn_input[nn_index]);
                bp.updateInputWeights(nn_history_input[i]);
                bp.savePrevWeights();
                bp.saveNextWeights();
            }
            writeHiddenWeights("hiddenWeights.txt");
            writeInputWeights("inputWeights.txt");

            // Track rewards
            this.totalReward += this.rewards;
            writeRewardRecord("rewardTracker.txt",this.totalReward);
        }
    }

    /** absoluteBearing: Computes the absolute bearing between two points */
    public double absoluteBearing(double x1, double y1, double x2, double y2){
        double x0 = x2-x1;
        double y0 = y2-y1;
        double hyp = Point2D.distance(x1,y1,x2,y2);
        double arcSin = Math.toDegrees(Math.asin(x0/hyp));
        double bearing = 0;

        if(x0>0 && y0 >0){
            // In the condition where both positions are in the lower-left
            bearing = arcSin;
        }
        else if(x0 < 0 && y0 > 0){
            // In the condition where we are in the lower-right
            bearing = 360 + arcSin;
        }
        else if(x0 > 0 && y0 < 0){
            // In the condition where we are in the upper-left
            bearing = 180-arcSin;
        }
        else if(x0 < 0 && y0 < 0){
            // In the condition where we are in the upper-right
            bearing = 180-arcSin;
        }
        return bearing;
    }

    /** Reading and saving LUT. Specified for two columns use */
    public void readLUT(String filename){
        List<String[]> dataList = new ArrayList<String[]>();
        try{
            BufferedReader readFile = new BufferedReader(new FileReader(getDataFile(filename)));
            Scanner reader = new Scanner(readFile);
            while(reader.hasNextLine()){
                String data = reader.nextLine();
                String[] parse = data.split("   ");
                dataList.add(parse);
            }
            System.out.println("FILE READ OK: File read successful!");
        } catch (FileNotFoundException e){
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

    /** initializeRecordFile: Get brand new files for recording wins and rewards */
    public void initializeRecordFile(String filename){
        try {
            RobocodeFileWriter f = new RobocodeFileWriter(getDataFile(filename));
            f.append("");
            f.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void readErrorRecord(String filename){
        try{
            BufferedReader reader = new BufferedReader(new FileReader(getDataFile(filename)));
            Scanner scanRead = new Scanner(reader);
            while (scanRead.hasNextLine()){
                String data = scanRead.nextLine();
                errorRecord = Double.parseDouble(data);
            }
            reader.close();
            scanRead.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeErrorRecord(String filename,double errorValue){
        try {
            RobocodeFileWriter f = new RobocodeFileWriter(getDataFile(filename));
            f.write(String.valueOf(errorRecord) + "\n");
            f.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void readRewardRecord(String filename){
        try{
            BufferedReader reader = new BufferedReader(new FileReader(getDataFile(filename)));
            Scanner scanRead = new Scanner(reader);
            while (scanRead.hasNextLine()){
                String data = scanRead.nextLine();
                this.rewardRecord.add(data);
            }
            reader.close();
            scanRead.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void readWinRecord(String filename){
        try{
            BufferedReader reader = new BufferedReader(new FileReader(getDataFile(filename)));
            Scanner scanRead = new Scanner(reader);
            while (scanRead.hasNextLine()){
                String data = scanRead.nextLine();
                this.winRecord.add(data);
            }
            reader.close();
            scanRead.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeRewardRecord(String filename,double rewardValue){
        this.rewardRecord.add(this.rewardRecord.size(),String.valueOf(rewardValue));
        try {
            RobocodeFileWriter f = new RobocodeFileWriter(getDataFile(filename));
            for(int i = 0;i<this.rewardRecord.size();i++) f.write(this.rewardRecord.get(i) + "\n");
            f.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /** onWin and onDeath would call this to record the win of the round
     * @param win 1 when the round is won, 0 if the round is lost */
    public void writeWinRecord(String filename, int win){
        this.winRecord.add(this.winRecord.size(),String.valueOf(win));
        try {
            RobocodeFileWriter f = new RobocodeFileWriter(getDataFile(filename));
            for(int i = 0;i<this.winRecord.size();i++) f.write(this.winRecord.get(i) + "\n");
            f.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /** writeHiddenWeights: Writes hidden weights into a file */
    public void writeHiddenWeights(String hidden_filename){
        double[] hiddenWeights = bp.getHiddenWeights();
        String[] hiddenString = new String[hiddenWeights.length];

        for(int i = 0; i< hiddenString.length; i++) hiddenString[i] = String.valueOf(hiddenWeights[i]);

        try{
            RobocodeFileWriter writeFile = new RobocodeFileWriter(getDataFile(hidden_filename));
            for(int i = 0;i<hiddenString.length;i++){
                writeFile.write(hiddenString[i] + "\n");
            }
            writeFile.close();
            System.out.println("FILE SAVED: Successfully saved!");
        } catch (IOException e) {
            System.out.println("FILE SAVE ERROR: Error in writing file");
            e.printStackTrace();
        }
    }

    /** writeInputWeights: Writes input weights into a file */
    public void writeInputWeights(String input_filename){
        double[] inputWeights = bp.getInputWeights();
        String[] inputString = new String[inputWeights.length];

        for(int i = 0; i< inputString.length; i++) inputString[i] = String.valueOf(inputWeights[i]);

        try{
            RobocodeFileWriter writeFile1 = new RobocodeFileWriter(getDataFile(input_filename));
            for(int i = 0;i<inputString.length;i++){
                writeFile1.write(inputString[i] + "\n");
            }
            writeFile1.close();
            System.out.println("FILE SAVED: Successfully saved!");
        } catch (IOException e) {
            System.out.println("FILE SAVE ERROR: Error in writing file");
            e.printStackTrace();
        }
    }

    /** readHiddenWeights: Reads the saved hidden weights from file and puts them back in BackPropagation.java */
    public void readHiddenWeights(String hidden_filename){
        ArrayList<String> hiddenArray = new ArrayList<String>();

        try{
            BufferedReader reader = new BufferedReader(new FileReader(getDataFile(hidden_filename)));
            Scanner scanRead = new Scanner(reader);
            while (scanRead.hasNextLine()){
                String data = scanRead.nextLine();
                hiddenArray.add(data);
            }
            reader.close();
            scanRead.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(int i = 0; i<hiddenArray.size();i++) bp.setHiddenWeights(Double.parseDouble(hiddenArray.get(i)),i);
    }

    /** readInputWeights: Reads the saved input weights from file and puts them back in BackPropagation.java */
    public void readInputWeights(String input_filename){
        ArrayList<String> inputArray = new ArrayList<String>();

        try{
            BufferedReader reader1 = new BufferedReader(new FileReader(getDataFile(input_filename)));
            Scanner scanRead1 = new Scanner(reader1);
            while(scanRead1.hasNextLine()){
                String data1 = scanRead1.nextLine();
                inputArray.add(data1);
            }
            reader1.close();
            scanRead1.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(int i = 0; i<inputArray.size();i++) bp.setInputWeights(Double.parseDouble(inputArray.get(i)),i);
    }

    /** Quantizing state to reduce LUT dimensionality */
    public void quantize(){
        this.quantizedX = quantizeX(this.robotXPos);
        this.quantizedY = quantizeY(this.robotYPos);
        this.quantizedDistance = quantizeDistance(this.robotDistToEnemy);
        this.quantizedBearing = quantizeBearing(this.robotAbsoluteBearing);
        System.out.println("ROBOT CURRENT STATE: " + this.quantizedX + this.quantizedY +
                this.quantizedDistance + this.quantizedBearing);
    }
    public int quantizeX(double x){
        int quantizedX = 1;
        if((x >= 0) && (x <=100)) quantizedX = 1;
        else if((x > 100) && (x <=200)) quantizedX = 2;
        else if((x > 200) && (x <=300)) quantizedX = 3;
        else if((x > 300) && (x <=400)) quantizedX = 4;
        else if((x > 400) && (x <=500)) quantizedX = 5;
        else if((x > 500) && (x <=600)) quantizedX = 6;
        else if((x > 600) && (x <=700)) quantizedX = 7;
        else if((x > 700) && (x <=800)) quantizedX = 8;
        return quantizedX;
    }
    public int quantizeY(double y){
        int quantizedY = 1;
        if((y >= 0) && (y <=100)) quantizedY = 1;
        else if((y > 100) && (y <=200)) quantizedY = 2;
        else if((y > 200) && (y <=300)) quantizedY = 3;
        else if((y > 300) && (y <=400)) quantizedY = 4;
        else if((y > 400) && (y <=500)) quantizedY = 5;
        else if((y > 500) && (y <=600)) quantizedY = 6;
        return quantizedY;
    }
    public int quantizeDistance(double dist){
        int quantizedDist = 1;
        if((dist >= 0) && (dist <= 250)) quantizedDist = 1;
        else if((dist > 250) && (dist <= 500)) quantizedDist = 2;
        else if((dist > 500) && (dist <= 750)) quantizedDist = 3;
        else if((dist > 750) && (dist <= 1000)) quantizedDist = 4;
        return quantizedDist;
    }
    public int quantizeBearing(double angle){
        int quantizedAngle = 1;
        if((angle > 0) && (angle <= 90)) quantizedAngle = 1;
        else if((angle > 90) && (angle <= 180)) quantizedAngle = 2;
        else if((angle > 180) && (angle <= 270)) quantizedAngle = 3;
        else if((angle > 270) && (angle <= 360)) quantizedAngle = 4;
        return quantizedAngle;
    }

}
