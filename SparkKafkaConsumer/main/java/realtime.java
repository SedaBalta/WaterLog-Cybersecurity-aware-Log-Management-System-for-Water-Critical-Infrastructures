import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.VoidFunction2;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.Trigger;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;
import static org.apache.spark.sql.functions.*;


import java.util.concurrent.TimeoutException;

import static org.apache.spark.sql.functions.*;

public class realtime {


     public static void myCustomFunc(Dataset<Row> df, long batchID) {
        // Prediction column= "0" topic name color


         Dataset<Row> normalDF = df.filter(functions.col("prediction").equalTo(0.0));
        Dataset<Row> normalDF2 = normalDF.withColumn("value",
                functions.concat(
                        functions.col("Treat_ScenarioID"), functions.lit(","),
                        functions.col("label"), functions.lit(","),
                        functions.col("prediction")
                ));

        normalDF2.select("value").write()
                .format("kafka")
                //.option("kafka.bootstrap.servers", "192.168.50.101:9092")
                .option("kafka.bootstrap.servers", "localhost:9092")
                //.option("topic", "normalRealtime")
                .option("topic", "normalRealtime2")
                .save();

        Dataset<Row> anormalDF = df.filter(functions.col("prediction").equalTo(1.0));
        Dataset<Row> anormalDF2 = anormalDF.withColumn("value",
                functions.concat(
                        functions.col("Treat_ScenarioID"), functions.lit(","),
                        functions.col("label"), functions.lit(","),
                        functions.col("prediction")
                ));

        anormalDF2.select("value").write()
                .format("kafka")
                //.option("kafka.bootstrap.servers", "192.168.50.101:9092")
                .option("kafka.bootstrap.servers", "localhost:9092")
                //.option("topic", "anormalRealtime")
                .option("topic", "anormalRealtime2")
                .save();
    }

    public static void main(String[] args) throws Exception/*, InterruptedException, StreamingQueryException, TimeoutException*/{

        Logger.getLogger("org").setLevel(Level.ERROR);
        SparkSession spark = SparkSession.builder()
                .appName("realtimeDeneme")
                .master("local[2]")
                .config("spark.driver.memory", "2g")
                .config("spark.executor.memory", "4g")
                .getOrCreate();

        //String filePath = "/Users/sbalta/Desktop/myWorks/SparkKafka/dataset2.json";
        String modelPath = "/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/modelnew";
        String checkPointDir = "/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/checkpoints/newcheckpoint9";

        StructType schema = new StructType()
                .add("Treat_ScenarioID", DataTypes.DoubleType)
                .add("Treat_Level2", DataTypes.DoubleType)
                .add("Treat_Level1", DataTypes.DoubleType)
                .add("Treat_CurrentVolume", DataTypes.DoubleType)
                .add("Dam_Level", DataTypes.DoubleType)
                .add("Dam_CurrentVolume", DataTypes.DoubleType)
                .add("Treat_PumpFlowLiterMin", DataTypes.DoubleType)
                .add("Treat_Pump3", DataTypes.StringType)
                .add("Treat_Pump2",DataTypes.StringType)
                .add("Treat_Pump1", DataTypes.StringType)
                .add("Treat_LimitSwitch1", DataTypes.StringType)
                .add("Dam_Pump3", DataTypes.StringType)
                .add("Dam_Pump2", DataTypes.StringType)
                .add("Dam_Pump1", DataTypes.StringType)
                .add("Dam_LimitSwitch", DataTypes.StringType)
                .add("Dam_Chlor_Raw", DataTypes.DoubleType)
                .add("label",DataTypes.DoubleType);

        Dataset<Row> data = spark.readStream()
                .format("kafka")
                //.option("kafka.bootstrap.servers", "192.168.50.101:9092")
                .option("kafka.bootstrap.servers", "localhost:9092")
                //.option("subscribe", "testRealtime")
                .option("subscribe", "jsonproducerp3")
                .load()
                .selectExpr("CAST(value AS STRING)");

        // Parse JSON data and apply schema
        Dataset<Row> jsonDataDF = data.select(from_json(col("value"), schema).as("data")).select("data.*");


        System.out.println("jsondataDF");
        jsonDataDF.printSchema();

        // Boolean sütununu one-hot encoding'e tabi tut
        Dataset<Row> encodedDF = jsonDataDF
                .withColumn("Treat_Pump3Index", when(col("Treat_Pump3").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Treat_Pump2Index", when(col("Treat_Pump2").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Treat_Pump1Index", when(col("Treat_Pump1").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Treat_LimitSwitch1Index", when(col("Treat_LimitSwitch1").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Dam_Pump3Index", when(col("Dam_Pump3").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Dam_Pump2Index", when(col("Dam_Pump2").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Dam_Pump1Index", when(col("Dam_Pump1").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Dam_LimitSwitchIndex", when(col("Dam_LimitSwitch").equalTo("true"), 1.0).otherwise(0.0));
        System.out.println("encodedDF tamamlandı.");
        System.out.println("vectorAssembler giris yapılıyor.");

        // Vector Assembler basladı.
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"Treat_Level2", "Treat_Level1", "Treat_CurrentVolume", "Dam_Level", "Dam_CurrentVolume", "Treat_PumpFlowLiterMin", "Dam_Chlor_Raw",
                        "Treat_Pump3Index", "Treat_Pump2Index", "Treat_Pump1Index", "Treat_LimitSwitch1Index", "Dam_Pump3Index", "Dam_Pump2Index", "Dam_Pump1Index",
                        "Dam_LimitSwitchIndex"})
                .setOutputCol("features")
                .setHandleInvalid("keep");

        System.out.println("assembler olusturuldu.");

        Dataset<Row> vectorAssemblerDF = assembler.transform(encodedDF);
        System.out.println("vectorassemblerDF olusturuldu");

        PipelineModel pipeModel = PipelineModel.load(modelPath);
        System.out.println("pipeline asamaları baslatıldı.");
        LogisticRegressionModel streamModel = (LogisticRegressionModel) pipeModel.stages()[1];
        System.out.println("logisticreg üzerinden gecirildi");

        Dataset<Row> transformedDF = streamModel.transform(vectorAssemblerDF);
        System.out.println("vectorassembler stream model üzerinden gecirildi.");
        transformedDF.printSchema();

        StreamingQuery query = transformedDF.writeStream()
                .foreachBatch((VoidFunction2<Dataset<Row>, Long>) realtime::myCustomFunc)
                .option("checkpointLocation", checkPointDir)
                .option("failOnDataLoss", "false")
                .trigger(Trigger.ProcessingTime("1 seconds"))
                .start();
        //query.awaitTermination();

        try {
            query.awaitTermination();
        } catch (Exception e) {
            e.printStackTrace();
        }


/*
        StreamingQuery query =transformedDF
                .writeStream()
                .format("console")
                .outputMode("append")
                .trigger(Trigger.ProcessingTime("1 seconds"))
                .start();

        query.awaitTermination();*/
    }

}
