package mlModels.waterlog;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import java.io.IOException;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
public class sw_decision_tree {
    private static final String MODEL_PATH="/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/ml-models/waterlog/decisiontree";
    public static void main(String[] args) throws IOException{


        StructType schema = new StructType()
                .add("FIT101", DataTypes.DoubleType)
                .add("LIT101", DataTypes.DoubleType)
                .add("MV101", DataTypes.DoubleType)
                .add("P101", DataTypes.DoubleType)
                .add("P102", DataTypes.DoubleType)
                .add("AIT201", DataTypes.DoubleType)
                .add("AIT202", DataTypes.DoubleType)
                .add("AIT203", DataTypes.DoubleType)
                .add("FIT201", DataTypes.DoubleType)
                .add("MV201", DataTypes.DoubleType)
                .add("P201", DataTypes.DoubleType)
                .add("P202", DataTypes.DoubleType)
                .add("P203", DataTypes.DoubleType)
                .add("P204", DataTypes.DoubleType)
                .add("P205", DataTypes.DoubleType)
                .add("P206", DataTypes.DoubleType)
                .add("DPIT301", DataTypes.DoubleType)
                .add("FIT301", DataTypes.DoubleType)
                .add("LIT301", DataTypes.DoubleType)
                .add("MV301", DataTypes.DoubleType)
                .add("MV302", DataTypes.DoubleType)
                .add("MV303", DataTypes.DoubleType)
                .add("MV304", DataTypes.DoubleType)
                .add("P301", DataTypes.DoubleType)
                .add("P302", DataTypes.DoubleType)
                .add("AIT401", DataTypes.DoubleType)
                .add("AIT402", DataTypes.DoubleType)
                .add("FIT401", DataTypes.DoubleType)
                .add("LIT401", DataTypes.DoubleType)
                .add("P401", DataTypes.DoubleType)
                .add("P402", DataTypes.DoubleType)
                .add("P403", DataTypes.DoubleType)
                .add("P404", DataTypes.DoubleType)
                .add("UV401", DataTypes.DoubleType)
                .add("AIT501", DataTypes.DoubleType)
                .add("AIT502", DataTypes.DoubleType)
                .add("AIT503", DataTypes.DoubleType)
                .add("AIT504", DataTypes.DoubleType)
                .add("FIT501", DataTypes.DoubleType)
                .add("FIT502", DataTypes.DoubleType)
                .add("FIT503", DataTypes.DoubleType)
                .add("FIT504", DataTypes.DoubleType)
                .add("P501", DataTypes.DoubleType)
                .add("P502", DataTypes.DoubleType)
                .add("PIT501", DataTypes.DoubleType)
                .add("PIT502", DataTypes.DoubleType)
                .add("PIT503", DataTypes.DoubleType)
                .add("FIT601", DataTypes.DoubleType)
                .add("P601", DataTypes.DoubleType)
                .add("P602", DataTypes.DoubleType)
                .add("P603", DataTypes.DoubleType)
                .add("label", DataTypes.IntegerType);

        SparkSession spark = SparkSession
                .builder()
                .appName("batch")
                .config("spark.master", "local")
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> df=spark
                .read()
                .option("header",true)
                .format("csv")
                .schema(schema)
                .load("/home/waterquality/IdeaProjects/SparkKafkaConsumer (copy)/src/main/resources/S1.csv");


          // Sonuçları göster
        df.printSchema();
        df.show();
        // Eksik değerleri doldurma veya kaldırma
        Dataset<Row> dfCleaned = df.na().drop();


        //,
      //  FIT101	LIT101	 MV101	P101	P102	 AIT201	AIT202	AIT203	FIT201	 MV201	 P201	 P202	P203	 P204	P205	P206	DPIT301	FIT301	LIT301	MV301	MV302	 MV303	MV304	P301	P302	AIT401	AIT402	FIT401	LIT401	P401	P402	P403	P404	UV401	AIT501	AIT502	AIT503	AIT504	FIT501	FIT502	FIT503	FIT504	P501	P502	PIT501	PIT502	PIT503	FIT601	P601	P602	P603	Normal/Attack
        // Assuming encodedDF is your DataFrame
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"MV201","LIT101",	 "MV101",	"P101",	"P102",	 "AIT201",	"AIT202","AIT203",	"FIT201",	 "MV201",	 "P201",	 "P202",	"P203",	 "P204",	"P205",	"P206",	"DPIT301",	"FIT301",	"LIT301",	"MV301",	"MV302",	 "MV303",	"MV304",	"P301",	"P302",	"AIT401",	"AIT402",	"FIT401",	"LIT401",	"P401",	"P402",	"P403",	"P404",	"UV401",	"AIT501",	"AIT502",	"AIT503",	"AIT504",	"FIT501",	"FIT502",	"FIT503",	"FIT504",	"P501",	"P502",	"PIT501",	"PIT502",	"PIT503",	"FIT601",	"P601",	"P602",	"P603"})
                .setOutputCol("features_vector");

        // Decision Tree sınıflandırıcı tanımlama
        DecisionTreeClassifier dtClassifier = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features_vector")
                .setMaxDepth(5);

        // Pipeline aşamaları
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, dtClassifier});

        // Veri setini eğitim ve test olarak ayırma
        Dataset<Row>[] splits = df.randomSplit(new double[]{0.8, 0.2}, 1234L);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Modeli eğitme
        PipelineModel model = pipeline.fit(trainingData);

        // Modeli kaydetme
        model.write().overwrite().save(MODEL_PATH);
        System.out.println("Model saved at: " + MODEL_PATH);

        // Test verileri ile tahmin
        Dataset<Row> predictions = model.transform(testData);

        // Tahmin sonuçlarını göster
        predictions.select("features_vector", "label", "prediction").show();


        // Performans değerlendirmesi başladı.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy: " + accuracy);

        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = f1Evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1Score);

        MulticlassClassificationEvaluator precisionEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedPrecision");
        double precision = precisionEvaluator.evaluate(predictions);
        System.out.println("Precision: " + precision);

        MulticlassClassificationEvaluator recallEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedRecall");
        double recall = recallEvaluator.evaluate(predictions);
        System.out.println("Recall: " + recall);

    }
}
