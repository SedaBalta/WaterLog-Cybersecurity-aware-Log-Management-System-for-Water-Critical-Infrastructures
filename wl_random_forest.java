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
import org.apache.spark.sql.types.StructType;
import java.io.IOException;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.ml.classification.RandomForestClassifier;
public class wl_random_forest {
    private static final String MODEL_PATH="/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/ml-models/waterlog/randomforest";
    public static void main(String[] args) throws IOException{
         //DL,DP,TL,TF,TP,EL,EF,EP,CL,CPF,CVF,CP,CV,TKL,TVF,TV,Normal/Attack
        StructType schema = new StructType()
                .add("DL", DataTypes.DoubleType)
                .add("DP", DataTypes.DoubleType)
                .add("TL", DataTypes.DoubleType)
                .add("TF", DataTypes.DoubleType)
                .add("TP", DataTypes.DoubleType)
                .add("EL", DataTypes.DoubleType)
                .add("EF", DataTypes.DoubleType)
                .add("EP", DataTypes.DoubleType)
                .add("CL", DataTypes.DoubleType)
                .add("CPF", DataTypes.DoubleType)
                .add("CVF", DataTypes.DoubleType)
                .add("CP", DataTypes.DoubleType)
                .add("CV", DataTypes.DoubleType)
                .add("TKL", DataTypes.DoubleType)
                .add("TVF", DataTypes.DoubleType)
                .add("TV", DataTypes.DoubleType)
                .add("label",DataTypes.StringType);

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
                .load("/home/waterquality/IdeaProjects/SparkKafkaConsumer/src/main/resources/Dataset.csv");

        df.printSchema();
        df.show();

        // Eksik değerleri doldurma veya kaldırma
        Dataset<Row> dfCleaned = df.na().drop();

        // assuming df_raw is your DataFrame
        Dataset<Row> encodedDF = df
                .withColumn("label_index", when(col("label").equalTo("Normal"), 0.0).otherwise(1.0));

        System.out.println("encodedDF");
        encodedDF.show();

        // Assuming encodedDF is your DataFrame
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"DL","DP","TL","TF","TP","EL","EF","EP","CL","CPF","CVF","CP","CV","TKL","TVF","TV"})
                .setOutputCol("features");

        // Assuming encodedDF is your DataFrame
        Dataset<Row> vectorAssemblerDF = assembler.transform(encodedDF);
        System.out.println("vectorAssemblerDF");
        vectorAssemblerDF.show();
        vectorAssemblerDF.printSchema();

        RandomForestClassifier rfClassifier = new RandomForestClassifier()
                .setLabelCol("label_index")
                .setFeaturesCol("features")
                .setNumTrees(2);

        //multi kullanabilmek için .setfamily("multinominal") yapılmalı.https://spark.apache.org/docs/3.0.1/ml-classification-regression.html

        // Pipeline işlemleri başladı.
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, rfClassifier});
        PipelineModel model = pipeline.fit(encodedDF);

        // Modeli kaydet
        model.write().overwrite().save(MODEL_PATH);
        System.out.println("Saved");

        Dataset<Row> resultDF = model.transform(encodedDF);
        // Pipeline işlemleri tamamlandı.

        // Sonuçları göster
        System.out.println("resultDF:");
        resultDF.show();
        resultDF.printSchema();

        // Performans değerlendirmesi başladı.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label_index")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(resultDF);
        System.out.println("Accuracy: " + accuracy);

        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label_index")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = f1Evaluator.evaluate(resultDF);
        System.out.println("F1 Score: " + f1Score);

        MulticlassClassificationEvaluator precisionEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label_index")
                .setPredictionCol("prediction")
                .setMetricName("weightedPrecision");
        double precision = precisionEvaluator.evaluate(resultDF);
        System.out.println("Precision: " + precision);

        MulticlassClassificationEvaluator recallEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label_index")
                .setPredictionCol("prediction")
                .setMetricName("weightedRecall");
        double recall = recallEvaluator.evaluate(resultDF);
        System.out.println("Recall: " + recall);

    }
}
