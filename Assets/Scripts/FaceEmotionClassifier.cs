using UnityEngine;

//this is my first time coding for Unity and something like a Meta Quest, so there is probably going to be a lot of errors in the beginning
//what im doing for this deliverable is reading the Quest Pro face tracking data every frame and figure out which of the 7 emotions the user is making, using simple threshold rules (citing EM-FACS)
//this is the W5 version at the moment, im thinking to swap in a machine learning model later for W6 (need to bring it up in the weekly meeting)
public class FaceEmotionClassifier:MonoBehaviour
{
  //this is the Meta component that gives the 63 FACS values
  //we drag the object that has OVRFaceExpressions on it into this slot in the Unity inspector
  [SerializeField] private OVRFaceExpressions faceExpr;

  //this string is so other scripts like the UI display can read the current emotion
  public string CurrentEmotion{get; private set;} = "Neutral";

  //how strong a blendshape has to be like between 0 and 1, before we count it as "active"
  //i made this public so we can tune it in the inspector without editing code at the lab (hopefully if im not mistaken)
  [SerializeField] private float activationThreshold = 0.3f;

  //update runs once every frame, which is exactly what we want for real time detection
  void Update()
  {
    //if the headset has not given us valid face data yet, do nothing this frame
    //(i think this also avoids something called InvalidOperationException that happens if you read too early)
    if(faceExpr == null || !faceExpr.ValidExpressions)
      return;

    //grab the blendshape values we care about once, so the rules below (took me a while) are easy to read
    //these names come straight from the Meta OVRFaceExpressions.FaceExpression enum
    float innerBrowRaiserL = faceExpr[OVRFaceExpressions.FaceExpression.InnerBrowRaiserL];
    float innerBrowRaiserR = faceExpr[OVRFaceExpressions.FaceExpression.InnerBrowRaiserR];
    float outerBrowRaiserL = faceExpr[OVRFaceExpressions.FaceExpression.OuterBrowRaiserL];
    float outerBrowRaiserR = faceExpr[OVRFaceExpressions.FaceExpression.OuterBrowRaiserR];
    float browLowererL = faceExpr[OVRFaceExpressions.FaceExpression.BrowLowererL];
    float browLowererR = faceExpr[OVRFaceExpressions.FaceExpression.BrowLowererR];
    float jawDrop = faceExpr[OVRFaceExpressions.FaceExpression.JawDrop];
    float lipCornerPullerL = faceExpr[OVRFaceExpressions.FaceExpression.LipCornerPullerL];
    float lipCornerPullerR = faceExpr[OVRFaceExpressions.FaceExpression.LipCornerPullerR];
    float lipCornerDepressorL = faceExpr[OVRFaceExpressions.FaceExpression.LipCornerDepressorL];
    float lipCornerDepressorR = faceExpr[OVRFaceExpressions.FaceExpression.LipCornerDepressorR];
    float lipStretcherL = faceExpr[OVRFaceExpressions.FaceExpression.LipStretcherL];
    float lipStretcherR = faceExpr[OVRFaceExpressions.FaceExpression.LipStretcherR];
    float upperLipRaiserL = faceExpr[OVRFaceExpressions.FaceExpression.UpperLipRaiserL];
    float upperLipRaiserR = faceExpr[OVRFaceExpressions.FaceExpression.UpperLipRaiserR];
    float noseWrinklerL = faceExpr[OVRFaceExpressions.FaceExpression.NoseWrinklerL];
    float noseWrinklerR = faceExpr[OVRFaceExpressions.FaceExpression.NoseWrinklerR];

    //this is something im trying which i might remove, but i average the left and right sides so the rules read more cleanly
    //my reason is since most emotions are symmetric, so this is a fair simplification
    float browRaise = Average(innerBrowRaiserL, innerBrowRaiserR, outerBrowRaiserL, outerBrowRaiserR);
    float browLower = Average(browLowererL, browLowererR);
    float smile = Average(lipCornerPullerL, lipCornerPullerR);
    float frown = Average(lipCornerDepressorL, lipCornerDepressorR);
    float lipStretch = Average(lipStretcherL, lipStretcherR);
    float upperLip = Average(upperLipRaiserL, upperLipRaiserR);
    float noseWrinkle = Average(noseWrinklerL, noseWrinklerR);

    //now for the actual EM-FACS rules, each emotion is a combination of action units as we discussed in our meetings
    //i check from the most specific to the least specific so the clearest match wins
    //these AU combinations come from Ekman and Friesens EM-FACS (cited again)
    string detected = "Neutral";

    //starting with the hardest one, contempt, as its a bit of a tricky asymmetric one (one lip corner pulls, the other doesnt)
    //check the gap between the two sides. if one side pulls way more than the other, its contempt
    if(Mathf.Abs(lipCornerPullerL - lipCornerPullerR) > activationThreshold && smile < activationThreshold)
      detected = "Contempt";

    //for happiness, its both lip corners pulled up
    else if(smile > activationThreshold)
      detected = "Happiness";

    //for surprise, both brows go up and the jaw drops open
    else if(browRaise > activationThreshold && jawDrop > activationThreshold)
      detected = "Surprise";

    //for fear, brows raise and lips stretch sideways (i think thisll be tricky since its close to surprise, but the lips stretch instead of jaw drop)
    else if(browRaise > activationThreshold && lipStretch > activationThreshold)
      detected = "Fear";

    //for anger, brows pull down and inward
    else if(browLower > activationThreshold)
      detected = "Anger";

    //for disgust, nose wrinkles and the upper lip raises
    else if(noseWrinkle > activationThreshold && upperLip > activationThreshold)
      detected = "Disgust";

    //lastly, for sadness, the inner brows rais and lip corners pull down
    else if(Average(innerBrowRaiserL, innerBrowRaiserR) > activationThreshold && frown > activationThreshold)
      detected = "Sadness";

    //store the result so the UI script can show it on screen
    CurrentEmotion = detected;
  }

//tiny helper so i do not repeat the averaging math everywhere. takes any number of values
    private float Average(params float[] values)
    {
        float sum = 0f;
        foreach (float v in values)
        {
            sum += v;  
        }
        return sum / values.Length;
    }
}
