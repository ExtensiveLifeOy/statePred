using Microsoft.ML.Runtime.Api;

// Jitai data is feeback-19.csv used to predict the state of user based on adherence
// id,feedback_date,feeling,motivational_message,patient_message,Adherence
namespace statePred
{
    
       public class HeliData 
        {
       // date
        [Column("0")]
        public string FeedbackDate;
        
        // feeling
        [Column("1")]
        [ColumnName("Label")]
        public string Label;

        // motivational_message
        [Column("2")]
        public string MotivationalMessage;

        // participant message
        [Column("3")]
        public string UserPerception;

         [Column("4")]
        public float Adherence;
    }

    // HeliPrediction is the result returned from prediction operations       
    public class HeliPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}