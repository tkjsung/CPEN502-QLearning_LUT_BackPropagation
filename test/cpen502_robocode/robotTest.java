package cpen502_robocode;


import org.junit.Test;

import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class robotTest {
    /** LUT variables and number of states defined here */
//    final int x_state = 8;
//    final int y_state = 6;
//    final int dist_state = 4;
//    final int energy_state = 4;
//    final int actions = 4;
//    String[] statesLUT;

    String[][] LUT;


    /** Tests... Do Not Use. */
    /*
    public void quantizeTest(){
        double x = 700.134;
        double y = 540.51235;
        double dist = 811.13;
        double erg = 61.5;
        int x_quantized = quantizeX(x);
        int y_quantized = quantizeY(y);
        int dist_quantized = quantizeDistance(dist);
        int erg_quantized = quantizeEnergy(erg);
        System.out.println("X, Y, Dist, Erg: " + x_quantized+y_quantized+dist_quantized+erg_quantized);

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
    public int quantizeEnergy(double erg){
        int quantizedErg = 4;
        if((erg >= 0) && (erg <= 25)) quantizedErg = 1;
        else if((erg > 25) && (erg <= 50)) quantizedErg = 2;
        else if((erg > 50) && (erg <= 75)) quantizedErg = 3;
        else if(erg > 75) quantizedErg = 4;
        return quantizedErg;
    }

    // LUT Initialization... deprecated... all written in other functions.
    public void initializeLUT(){
        createLUT();
        System.out.println("This is LUT length: " + LUT.length);
        for(int i = 0; i<LUT.length; i++) LUT[i][1] = Double.toString(0.0);
        initializeLUTStates();
        System.out.println("LUT states initialized.");
    }
    public void initializeLUTStates(){
        int counter = 0;
        for(int i = 1; i<=x_state;i++){
            for(int j = 1; j<=y_state;j++){
                for(int k = 1; k<=dist_state;k++){
                    for(int l = 1;l<=energy_state;l++){
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
    }

    public void LUTSaveTest(){
        createLUT();
        initializeLUT();
        writeLUT(LUT,"LUT.txt");
        LUT = readLUT("LUT.txt");
        System.out.println("LUT save test completed.");
    }
*/


    public void testBP(){
        algorithmBP a = new algorithmBP();
        a.setBiasWeight(true);
        a.setIterations(1);
        a.setLearnRate(0.00005);
        a.setMomentum(0.9);
        a.setRepresentation(true);
        a.readLUT("/Users/tomsung/IdeaProjects/CPEN502_Robocode" +
                "/out/production/CPEN502_Robocode/cpen502_robocode/MyRobot_BP_20201127.data/LUT.txt");
        a.LUTTrainingSet();
        a.algorithm();
        System.out.println("Test");
    }

    public void testInt(){
        int j = 11423;
        String k = String.valueOf(j);
        String[] t = k.split("");
        int m = Integer.parseInt(t[4]);
        System.out.println("End of program");
//        char l = k.charAt(4);
//        String m = String.valueOf(l);

    }
    public void randomNumberAction(){
        int upperbound = 4;
        Random rng = new Random();
        int randomNum = rng.nextInt(upperbound) + 1;
        System.out.println(randomNum);
//        return randomNum;
    }

    public void stringjoin(){
        String[] news = new String[3];
        news[0] = "2";
        news[1] = "1";
        news[2] = "5";
        String joined = String.join("",news);
        System.out.println(joined);
    }


    public void writeAppend(){
        try {
            ArrayList<String> winRecord = new ArrayList<String>();
            FileReader reader = new FileReader("test.txt");
            Scanner scanRead = new Scanner(reader);
            while (scanRead.hasNextLine()){
                String data = scanRead.nextLine();
                winRecord.add(data);
            }
            winRecord.add(winRecord.size(),String.valueOf(3));
//            FileWriter f = new FileWriter("test.txt",true);
//            BufferedWriter b = new BufferedWriter(f);
//            PrintWriter p = new PrintWriter(b);
            FileWriter f = new FileWriter("test.txt");
            for(int i = 0;i< winRecord.size();i++) f.write(winRecord.get(i) + "\n");
//            p.println("Test123");
//            p.println(1);
//            p.println(-24.13);
//            p.close();
//            b.close();
            reader.close();
            f.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void array2D(){
        ArrayList<String[]> row  = new ArrayList<>();
        String[] column = new String[3];
        column[0] = "test";
        row.add(0,column);
        column[0] = "12";
        row.add(1,column);
        System.out.println("Test");
    }

    public void testInteger(){
        ArrayList<String> test = new ArrayList<String>();
        test.add("1");
        test.add("4");
//        int x = 4;
//        double x = -0.00000000000;
//        double y = Double.valueOf(x);
//        System.out.println(x + " and " + y);
        System.out.println(Double.parseDouble(test.get(1)));
        System.out.println(test.get(1));
    }


    /** More previously written code for tests */

    /** Creating LUT */
  /*
    public void createLUT(){
        LUT = new String[x_state*y_state*dist_state*energy_state*actions][2]; //LUT array created
        //States array used to initialize states:
        statesLUT = new String[x_state*y_state*dist_state*energy_state*actions];
        System.out.println("LUT created.");
        System.out.println("This is LUT length: " + LUT.length);

        // Initialize values to 0
        for(int i = 0; i<LUT.length; i++) LUT[i][1] = Double.toString(0.0);
        // Initialize states
        int counter = 0;
        for(int i = 1; i<=x_state;i++){
            for(int j = 1; j<=y_state;j++){
                for(int k = 1; k<=dist_state;k++){
                    for(int l = 1;l<=energy_state;l++){
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
*/

    /** Initialize all LUT values to 0 */
    /*
    public void initializeLUT(){
        for(int i = 0; i<LUT.length; i++) LUT[i][1] = Double.toString(0.0);
    }
    */

    /** Reading and saving LUT */
    /*
    public void writeLUT(String[][] data, String filename){
        try{
            FileWriter writeFile = new FileWriter(filename);
            for(int i = 0;i<data.length;i++) writeFile.write(data[i][0] + "   " + data[i][1] + "\n");
            writeFile.close();
            System.out.println("FILE: Successfully saved!");
        } catch (IOException e){
            System.out.println("FILE: Error in writing file");
            e.printStackTrace();
        }
    }
    public String[][] readLUT(String filename){
        List<String[]> dataList = new ArrayList<String[]>();
        try{
            File readFile = new File(filename);
            Scanner reader = new Scanner(readFile);
            while(reader.hasNextLine()){
                String data = reader.nextLine();
                String[] parse = data.split("   ");
                dataList.add(parse);
            }
            System.out.println("FILE: File read successful!");
        } catch (FileNotFoundException e){
            System.out.println("FILE: File read error happened");
            e.printStackTrace();
        }

        String[][] data_output = new String[dataList.size()][2];
        for(int i = 0; i<dataList.size();i++){
            String[] parse;
            parse = dataList.get(i);
            data_output[i][0] = parse[0];
            data_output[i][1] = parse[1];

        }
        return data_output;
    }
*/


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
    public void fileDelete(String filename){
        File deleteFile = new File(filename);
        if (deleteFile.delete()){
            System.out.println("Deleted the file: " + deleteFile.getName());
        } else {
            System.out.println("Failed to delete the file.");
        }
    }


}
