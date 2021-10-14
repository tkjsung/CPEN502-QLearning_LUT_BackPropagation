/**
 * MyRobot_20201101.java - Reinforcement Learning with Look-Up Table (Immediate Rewards)
 * @author Tom (Ke-Jun) Sung
 * @since 2020-11-12
 */

package cpen502_robocode;

import robocode.*;
import robocode.RobocodeFileWriter;

import java.awt.geom.Point2D;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class MyRobot_20201101 extends AdvancedRobot{
    /** LUT variables and number of states defined here */
    final int x_state = 8;
    final int y_state = 6;
    final int dist_state = 4;
    final int energy_state = 4;
    final int bearing_state = 4;
    final int actions = 4;
    private String[][] LUT;
    private String[] statesLUT;

    /** Quantized values for LUT. Default values are given to avoid errors/warnings */
    private int quantizedX = 1;
    private int quantizedY = 1;
    private int quantizedDistance = 1;
    private int quantizedEnergy = 4;
    private int quantizedBearing = 4;

    /** Status of my robot */
    private double robotBearing = 0.0; // Get bearing. Initialized to 0.
    private double robotAbsoluteBearing = 0.0; // Get absolute bearing. Initialized to 0.
    private double robotHeading = 0.0; // Get heading. Initialized to 0.
    private double robotXPos = 0.0; // Get x-coordinate for current location. Initialized to 0.
    private double robotYPos = 0.0; // Get y-coordinate for current location. Initialized to 0.
    private double robotDistToEnemy = 0.0; // Get distance to enemy from current location. Initialized to 0.
    private double robotEnergy = 100.0; // Get energy of this robot. Initialized to 100.

    /** Current/Next state action values */
    private int currentState = 11141;
    private int nextState = 11141;
    private double currentStateValue = 0.0;
    private double nextStateValue = 0.0;

    /** Other required variables */
    String[][] selectedStates = new String[actions][2];
    private int initial = 0;
    private int hasRoundEnded = 0;
    private double rewards = 0;
    private double totalReward = 0;
    private ArrayList<String> winRecord = new ArrayList<String>();
    private ArrayList<String> rewardRecord = new ArrayList<String>();

    /** Variables that can be changed to exhibit different behaviour */
    final double stepSize = 0.1;
    final double gamma = 0.9;
    final double explore = 0.7;
    final boolean offPolicy = true;

    public void run() {
        // Due to how Robocode handles this object, we must read the LUT every time a battle starts.
        // Or alternatively, create a separate Java Object and import LUT with static.
        /**
         * The following three functions that are placed on the same line is used for the beginning of training.
         * Please uncomment to get a fresh new LUT file.
         * Best way: Toggle a breakpoint on readLUT(). Enter Debug mode.
         * Add only this tank into Robocode to save time.
         */
//        createLUT(); initializeLUT(); writeLUT(LUT,"LUT.txt");
//        initializeRecordFile("winCounter.txt"); initializeRecordFile("rewardTracker.txt");

        // Initializing state. (more like predicting initial state)
        readLUT("LUT.txt");
        readWinRecord("winCounter.txt");
        readRewardRecord("rewardTracker.txt");
        updateRobotStatus();
        turnGunRight(360); initial = 1; // Initialize the states by scanning the environment
        quantize();
        getSelectedStatesAction(); // Given the states, add actions to them and get their values.
        int maxState = 11141; double maxStateValue = -1000; // Initialized to avoid warnings

        // Take maximum of current state (epsilon-greedy).
        for(int i = 0; i<actions; i++){
            if(Double.parseDouble(this.selectedStates[i][1]) > maxStateValue){
                maxState = Integer.parseInt(this.selectedStates[i][0]);
                maxStateValue = Double.parseDouble(this.selectedStates[i][1]);
            }
        }
        // We finally now have the state-action pair, given the selected policy.
        this.currentState = maxState;
        this.currentStateValue = maxStateValue;

        while (true) {
            // Q-Learning algorithm must be written here. Otherwise, it wouldn't work
            // Please save and load the LUT table every time

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

                // Take maximum (for on-policy. For no exploration, this is required)
                maxState = 11141; maxStateValue = -1000; // Initialized to avoid warnings
                for(int i = 0; i<actions; i++){
                    if(Double.parseDouble(this.selectedStates[i][1]) > maxStateValue){
                        maxState = Integer.parseInt(this.selectedStates[i][0]);
                        maxStateValue = Double.parseDouble(this.selectedStates[i][1]);
                    }
                }
                this.nextState = maxState;
                this.nextStateValue = maxStateValue;

                // Now update states (Q equation).
                System.out.println("UPDATING Q-EQUATION");
                double tempStateValue = this.currentStateValue + this.stepSize*(this.rewards +
                        this.gamma*this.nextStateValue-this.currentStateValue);

                // Update LUT
                for(int i = 0; i<LUT.length; i++){
                    if(LUT[i][0].equals(String.valueOf(this.currentState))){
                        LUT[i][1] = String.valueOf(tempStateValue);
                    }
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
                    if(Integer.parseInt(this.LUT[i][0]) == this.currentState){
                        this.currentStateValue = Double.parseDouble(this.LUT[i][1]);
                    }
                }

                robotAction(actionDigit); // Perform action

                // Observe the environment and the rewards (Scan environment for next state)
                turnGunRight(360);
                updateRobotStatus();
                quantize();
                getSelectedStatesAction();

                // Get the maximum state-action pair (to ensure proper transition into non-explore state)
                maxState = 11141; maxStateValue = -1000; // Initialized to avoid warnings
                for(int i = 0; i<actions; i++){
                    if(Double.parseDouble(this.selectedStates[i][1]) > maxStateValue){
                        maxState = Integer.parseInt(this.selectedStates[i][0]);
                        maxStateValue = Double.parseDouble(this.selectedStates[i][1]);
                    }
                }
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

                // Update LUT
                for(int i = 0; i<LUT.length; i++){
                    if(LUT[i][0].equals(String.valueOf(this.currentState))){
                        LUT[i][1] = String.valueOf(tempStateValue);
                    }
                }
            } // end else loop for exploration

            // Save LUT. Save rewards. Save wins.
            // Assign next state to current state
            writeLUT(LUT,"LUT.txt");
            this.currentState = this.nextState;
            this.currentStateValue = this.nextStateValue;
            this.totalReward += this.rewards;


        } // end while()

    } // end run()


    /**
     * getSelectedStates: Given the quantized states, attach possible actions to them. Then put their values in a
     * matrix selectedStates[][]. This will help with either on- or off-policy situations. */
    public void getSelectedStatesAction(){
        for(int i = 0; i<this.actions; i++) {
            this.selectedStates[i][0] = Integer.toString(this.quantizedX) + Integer.toString(this.quantizedY) +
                    Integer.toString(this.quantizedDistance) + Integer.toString(this.quantizedBearing) +
                    Integer.toString(i+1);
        }
        for(int j = 0; j<this.actions; j++){
            for(int i = 0; i<this.LUT.length; i++){
                if(this.LUT[i][0].equals(this.selectedStates[j][0])){
                    this.selectedStates[j][1] = this.LUT[i][1];
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

    /** Random Number Generation Functions */
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
        this.robotEnergy = getEnergy();
    }

    /** onScannedRobot: What to do when you see another robot */
    public void onScannedRobot (ScannedRobotEvent e){
        this.robotBearing = e.getBearing(); // In angles
        this.robotDistToEnemy = e.getDistance();

        double angle = Math.toRadians((getHeading() + this.robotBearing % 360));
        double enemyX = (this.robotXPos + Math.sin(angle) * this.robotDistToEnemy);
        double enemyY = (this.robotYPos + Math.cos(angle) * this.robotDistToEnemy);
        this.robotAbsoluteBearing = absoluteBearing(getX(), getY(), enemyX, enemyY);


        // Shooting happens here. Now commented out
        /*
        if(this.initial != 0){
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
        } else {
            System.out.println("Performing initial system scan.");
        }
        */

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
            // Update LUT
            for(int i = 0; i<LUT.length; i++){
                if(LUT[i][0].equals(String.valueOf(this.currentState))){
                    LUT[i][1] = String.valueOf(tempStateValue);
                }
            }
            writeLUT(LUT,"LUT.txt");

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
            // Update LUT
            for(int i = 0; i<LUT.length; i++){
                if(LUT[i][0].equals(String.valueOf(this.currentState))){
                    LUT[i][1] = String.valueOf(tempStateValue);
                }
            }
            writeLUT(LUT,"LUT.txt");

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

    /** Creating LUT */
    public void createLUT(){
        this.LUT = new String[x_state*y_state*dist_state*bearing_state*actions][2]; //LUT array created
        //States array used to initialize states:
        this.statesLUT = new String[x_state*y_state*dist_state*bearing_state*actions];
        System.out.println("LUT created.");
        System.out.println("This is LUT length: " + LUT.length);

        // Initialize values to 0
        for(int i = 0; i<LUT.length; i++) LUT[i][1] = Double.toString(0.0);
        // Initialize states
        int counter = 0;
        for(int i = 1; i<=x_state;i++){
            for(int j = 1; j<=y_state;j++){
                for(int k = 1; k<=dist_state;k++){
                    for(int l = 1;l<=bearing_state;l++){
                        for(int m = 1; m<=actions; m++){
                            statesLUT[counter] = Integer.toString(i)+Integer.toString(j)+Integer.toString(k)+
                                    Integer.toString(l)+Integer.toString(m);
                            counter++;
                        }
                    }
                }
            }
        }
        for(int i = 0; i<LUT.length; i++) LUT[i][0] = statesLUT[i];
        System.out.println("LUT initialized.");
    }

    /** Initialize all LUT values to 0 */
    public void initializeLUT(){
        for(int i = 0;i<this.LUT.length;i++) this.LUT[i][1]=Double.toString(0.0);
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
        this.LUT = data_output;
    }
    public void writeLUT(String[][] data, String filename) {
        try{
            RobocodeFileWriter writeFile = new RobocodeFileWriter(getDataFile(filename));
            for(int i = 0;i<data.length;i++) writeFile.write(data[i][0] + "   " + data[i][1] + "\n");
            writeFile.close();
            System.out.println("FILE SAVED: Successfully saved!");
        } catch (IOException e) {
            System.out.println("FILE SAVE ERROR: Error in writing file");
            e.printStackTrace();
        }
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

    /** Quantizing state to reduce LUT dimensionality */
    public void quantize(){
        this.quantizedX = quantizeX(this.robotXPos);
        this.quantizedY = quantizeY(this.robotYPos);
        this.quantizedDistance = quantizeDistance(this.robotDistToEnemy);
        this.quantizedEnergy = quantizeEnergy(this.robotEnergy); // This line is not needed anymore
        this.quantizedBearing = quantizeBearing(this.robotAbsoluteBearing);
//        System.out.println("ROBOT CURRENT STATE: " + this.quantizedX + this.quantizedY +
//                this.quantizedDistance + this.quantizedEnergy);
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
    // quantizeEnergy no longer needed. Was used before.
    public int quantizeEnergy(double erg){
        int quantizedErg = 4;
        if((erg >= 0) && (erg <= 25)) quantizedErg = 1;
        else if((erg > 25) && (erg <= 50)) quantizedErg = 2;
        else if((erg > 50) && (erg <= 75)) quantizedErg = 3;
        else if(erg > 75) quantizedErg = 4;
        return quantizedErg;
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
