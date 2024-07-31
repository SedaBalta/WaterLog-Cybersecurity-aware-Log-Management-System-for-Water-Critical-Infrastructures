import org.apache.hadoop.shaded.com.nimbusds.jose.shaded.json.JSONArray;
import org.apache.hadoop.shaded.com.nimbusds.jose.shaded.json.parser.JSONParser;
import org.apache.hadoop.shaded.com.nimbusds.jose.shaded.json.parser.ParseException;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;
import org.apache.spark.sql.streaming.Trigger;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import scala.collection.Seq;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeoutException;

import static org.apache.spark.sql.execution.command.ClearCacheCommand.schema;
import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.types.DataTypes.IntegerType;
import static org.apache.spark.sql.types.DataTypes.StringType;

public class Main {
    private static final String MODEL_PATH="/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/model";

    public static void main(String[] args) throws TimeoutException, StreamingQueryException, ParseException {


        String KafkaTopic="sample";
        String KafkaTopic2="bizimtopic2";
        String KafkaBootstrapServers="localhost:9092";


        SparkSession spark= SparkSession
                .builder()
                .appName("JavaStructureStreaming")
                .master("local[*]")
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        PipelineModel pipelineModel=PipelineModel.load(MODEL_PATH);

        Dataset<Row> lines= spark
                .readStream()
                .format("kafka")
                .option("kafka.bootstrap.servers",KafkaBootstrapServers)
                .option("subscribe",KafkaTopic)
                .option("startingOffsets","earliest")
                .load().selectExpr( "CAST(key AS STRING)","CAST(value AS STRING)");


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

        Dataset<Row> output= lines.select(from_json(lines.col("value").cast("string"),DataType.fromJson(schema.json())).as("data")).select("data.*");
         //{"Treat_ScenarioID": 585, "Treat_Level2": 4, "Treat_Level1": 8, "Treat_CurrentVolume": 9976, "Dam_Level": 2, "Dam_CurrentVolume": 18961, "Treat_PumpFlowLiterMin": -0.373501851, "Treat_Pump3": false, "Treat_Pump2": true, "Treat_Pump1": true, "Treat_LimitSwitch1": true, "Dam_Pump3": true, "Dam_Pump2": true, "Dam_Pump1": true, "Dam_LimitSwitch": true, "Dam_Chlor_Raw": 90, "label": 1}

        output.printSchema();

   /*     List<String> columnsToIndex = Arrays.asList("Treat_Pump3","Treat_Pump2","Treat_Pump1","Treat_LimitSwitch1","Dam_Pump3","Dam_Pump2","Dam_Pump1","Dam_LimitSwitch");
*//*

        for (String columnName : columnsToIndex) {
            System.out.println(columnName);
        }
*//*

        for (String colName : columnsToIndex) {
            StringIndexerModel indexer = new StringIndexer()
                    .setInputCol(colName)
                    .setOutputCol(colName + "_index")
                    .fit(output);


            output = indexer.transform(output).drop(colName);

            // Indexlenmiş sütunu orijinal sütunun yerine yerleştirme

            output = output.withColumn(colName, col(colName + "_index")).drop(colName + "_index");
        }
        output.show();
        output.printSchema();
*/


/*
        VectorAssembler vectorAssembler=new VectorAssembler() //indextostring metodu ile
                .setInputCols(new String[]{"Treat_Level2","Treat_Level1","Treat_CurrentVolume","Dam_Level","Dam_CurrentVolume","Treat_PumpFlowLiterMin",
                        "Treat_Pump3","Treat_Pump2","Treat_Pump1","Treat_LimitSwitch1","Dam_Pump3","Dam_Pump2","Dam_Pump1","Dam_LimitSwitch"})
                .setOutputCol("features");

        Dataset<Row> output2 = vectorAssembler.transform(output);

        output2.show();
        output2.printSchema();
*/



        //  Dataset<Row> result=pipelineModel.transform(output);

    /*   BU KOD ŞEMASIZ JSON FORMATINDAKİ VERİYİ PARSE EDİYOR.
                Dataset<Row> dz = lines.select(
                from_json(lines.col("value"), DataTypes.createStructType(new StructField[] {DataTypes.createStructField("Name", StringType,true)})).getField("Name").alias("Name")
                ,from_json(lines.col("value"), DataTypes.createStructType(new StructField[] {DataTypes.createStructField("Age", IntegerType,true)})).getField("Age").alias("Age"));
*/
             /*   StreamingQuery query=lines
                .writeStream()
                .format("Kafka")
                .option("kafka.bootstrap.servers",KafkaBootstrapServers)
                .option("subscribe",KafkaTopic)
                .option("checkpointLocation","checkpointdd/")
                .start();
*/

        StreamingQuery query =output
                .writeStream()
                        .format("console")
                                .outputMode("append")
                                        .trigger(Trigger.ProcessingTime("1 seconds"))
                                                .start();

        query.awaitTermination();

    }
}