//this is the MLP version of my cosine classifier 
//it reads the same 63 face blendshapes off the Quest Pro but instead of doing cosine math it runs the pre-trained neural net (the one i converted to fau_model.onnx) right on the headset using Unity's Inference Engine
//almost everything here is reused from my cosine setup. the only new bit is the 4 lines that actually run the model (Schedule/PeekOutput/DownloadToArray)
//reading the blendshapes and showing a label is the same idea as FaceEmotionClassifier.cs + EmotionDisplay.cs

using UnityEngine;

//the new name for Sentis
//if package is older and this line errors change it to using Unity.Sentis;
using Unity.InferenceEngine;   
using TMPro;

public class MLPEmotionClassifier:MonoBehaviour
{
  //the imported fau_model.onnx
  public ModelAsset modelAsset;
  //same component my cosine script uses
  public OVRFaceExpressions faceExpressions;
  //optional on-screen label to read the prediction
  public TMP_Text label;                   

  //the result exposed the same way my cosine classifier exposes it so anything that read CurrentEmotion before can read it here too
  public string CurrentEmotion {get; private set;} = "...";

  //the 7 outputs in the EXACT order the model was trained (matches the python converter)
  //note this model has neutral and has no contempt unlike my cosine set
  static readonly string[] LABELS = {"Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"};

  //the inference engine that runs the model
  Worker worker;
  //scratch buffer for the 63 blendshape values
  readonly float[] faces = new float[63];  

  void Start()
  {
    //load the model and build a worker
    //CPU backend because this net is tiny so the CPU is more than fast enough and i avoid the GPU read-back stalls Sentis warns about
    var model = ModelLoader.Load(modelAsset);
      worker = new Worker(model, BackendType.CPU);

      //warm-up run once so the first real frame doesnt hitch (Metas docs recommend this)
      using (var warm = new Tensor<float>(new TensorShape(1, 63), faces))
      {
          worker.Schedule(warm);
          worker.PeekOutput();
      }
  }

  void Update()
  {
    //only run if the face tracking is actually giving us valid data this frame
    if(faceExpressions == null || !faceExpressions.FaceTrackingEnabled || !faceExpressions.ValidExpressions)
      return;

    //grab the 63 blendshapes in OVR enum order
    //the model was trained on the EmoHeVRDB columns which are in this same order so this lines up 
    //if the predictions ever look random, this ordering is the first thing to check
    for(int i = 0; i < (int)OVRFaceExpressions.FaceExpression.Max; i++)
      faces[i] = faceExpressions[(OVRFaceExpressions.FaceExpression)i];

    //make a [1, 63] input tensor from those values, run the model, copy the 7 scores back
    using(var input = new Tensor<float>(new TensorShape(1, 63), faces))
    {
      //run inference (non-blocking)
      worker.Schedule(input);
      //grab the 7-score output
      var output = worker.PeekOutput() as Tensor<float>;
      //pull it to the CPU so i can read it
      float[] scores = output.DownloadToArray();           

      //argmax so whichever of the 7 scores is highest is the predicted emotion
      int best = 0;
      for(int i = 1; i < scores.Length; i++)
        if(scores[i] > scores[best]) 
          best = i;

      CurrentEmotion = LABELS[best];
      //e.g. "Happy (0.82)"
      if(label != null) label.text = $"{CurrentEmotion} ({scores[best]:0.00})"; 
    }
  }

  void OnDisable()
  {
    //free the engine's memory when this object goes away
    worker?.Dispose();
  }
}
