using UnityEngine;
using TMPro;
using System.Collections.Generic;

//this replaces the debug display from w5/w6 with a clean ar overlay for w7
//the old version showed tracking status and the raw blendshape vector which was needed to verify the pipeline
//now that the pipeline is confirmed working we just want the emotion label visible in passthrough ar
//color coding approach taken from affective ar visualization research (semsioğlu & yantaç 2022 emote tool at ah2022)
//tmp rich text color tags documented at docs.unity3d.com/Packages/com.unity.textmeshpro
public class EmotionDisplay:MonoBehaviour
{
  //drag the object with FaceEmotionClassifier into this slot in the inspector
  [SerializeField] private FaceEmotionClassifier classifier;

  //drag the tmp text gameobject into this slot
  [SerializeField] private TextMeshProUGUI emotionText;

  private string lastEmotion = "";

  //each emotion maps to a hex color so the label pops visually against passthrough
  //colors follow standard affective associations (yellow=happy blue=sad red=anger etc)
  private static readonly Dictionary<string, string> emotionColors = new Dictionary<string, string>
  {
    {"Happiness", "#FFD700"},
    {"Sadness",   "#4488FF"},
    {"Surprise",  "#FF8800"},
    {"Fear",      "#CC44FF"},
    {"Anger",     "#FF3333"},
    {"Disgust",   "#44BB44"},
    {"Contempt",  "#AAAAAA"},
    {"Neutral",   "#FFFFFF"},
  };

  void Update()
  {
    if(classifier == null || emotionText == null)
      return;

    string current = classifier.CurrentEmotion;

    //only redraw the label when the emotion actually changes
    //no point calling tmp every frame for the same string
    if(current == lastEmotion)
      return;

    //fall back to white if somehow the emotion string isnt in the table
    string hex = emotionColors.ContainsKey(current) ? emotionColors[current] : "#FFFFFF";

    //tmp rich text: color tag + bold wrap the emotion name
    emotionText.text = $"<color={hex}><b>{current}</b></color>";
    lastEmotion = current;
  }
}
