using System;
using Microsoft.ML; // 0.1.0 at 9.5.2018 :-)
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace statePred
{
    class Program
    {
         // STEP 1: Define your data structures
        // use HeliData       
     
        static void Main(string[] args)
        {
            Console.WriteLine("ML for Heli state!");

            // STEP 2: Create a pipeline and load your data
            var pipeline = new LearningPipeline();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of dataPath is set to 'Copy always'
            string dataPath = "feedbacks-19.txt";
            //pipeline.Add(new TextLoader<IrisData>(dataPath, separator: ","));
            pipeline.Add(new TextLoader<HeliData>(dataPath, separator: ","));

            // STEP 3: Transform your data
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training
            pipeline.Add(new Dictionarizer("Label"));

            // Puts all features into a vector            
            pipeline.Add(new ColumnConcatenator("Features", "Adherence"));

            // STEP 4: Add learner
            // Add a learning algorithm to the pipeline. 
            // This is a classification scenario (Question: What type of state is this?)
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            //pipeline.Add(new NaiveBayesClassifier());
            //pipeline.Add(new LogisticRegressor());

             // Convert the Label back into original text (after converting to number in step 3)
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 5: Train your model based on the data set
            var model = pipeline.Train<HeliData, HeliPrediction>();

            // STEP 6: Use your model to make a prediction
            // You can change these numbers to test different predictions            
            var inputData = new HeliData()
            {
                Adherence = 0.3f                
            };
             var prediction = model.Predict(inputData);
            Console.WriteLine($"Predicted state type for adh {inputData.Adherence} is: {prediction.PredictedLabels}");
        }
    }
}
