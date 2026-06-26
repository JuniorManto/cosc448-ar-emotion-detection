using System.Linq;
using System;
using UnityEngine;

//this is an additional script to display the emotion
//right now, the classifier figures out the emotion internally, but there's nothing on screen to say what the emotion is (no visible output)
//so this script will put a basic text label in the headset view so the camera has something to capture

//i saw online people using this TMPro which is Unity's nicer text system and looks better than plain old text
using TMPro;

//like i said before, this script takes whatever emotion the classifier guessed and shows it as text in the headset
//it doesnt do any detecting itself, it only reads the emotion from FaceEmotionClassifier and displays it
public class EmotionDisplay:MonoBehaviour
{
  //drag the object that has the FaceEmotionClassifier script into this slot in the inspector
  [SerializeField] private FaceEmotionClassifier classifier;

  //drag the TMPro text object into this slot, which is the actual text we change on screen
  [SerializeField] private TextMeshProUGUI emotionText;

  //keep track of the last emotion we showed so we only update the text when it actually changes
  //no point in rewriting the same text every frame 
  private string lastEmotion = "";

  void Update()
  {
    //safety check. if i forgot to hook something up in the inspector, do nothing instead of crashing
    if(classifier == null || emotionText == null)
      return;
    try
    {
      //grab the current emotion from the classifier script
      string current = classifier.CurrentEmotion;

      //only bother updating the on screen text if the emotion changed from the previous frame
      if (current != lastEmotion)
      {
        string vecStr = classifier.CurrentFaceVector != null
          ? string.Join(", ", classifier.CurrentFaceVector.Select(f => f.ToString("F3")))
          : "null";

        bool trackingEnabled = classifier.FaceExpr != null && classifier.FaceExpr.FaceTrackingEnabled;

        emotionText.text = $"OVRFaceExpressions Status: {trackingEnabled};\n\nCurrent Emotion: {current ?? "null"}\n\nCurrent Face Data:\n[{vecStr}]";
        lastEmotion = current;
      }
    }
    catch(Exception e)
    {
      Debug.LogError($"[Update] Crashed: {e.Message}");
      Debug.Log($"FaceExpr null? {classifier.FaceExpr == null}");
      Debug.Log($"CurrentFaceVector null? {classifier.CurrentFaceVector == null}");
      Debug.Log($"CurrentEmotion null? {classifier.CurrentEmotion == null}");
    }
  }
}
