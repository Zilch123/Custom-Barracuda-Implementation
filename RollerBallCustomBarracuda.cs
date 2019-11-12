using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using Barracuda;
using System.IO;
using UnityEngine.UI;
using MLAgents.InferenceBrain;



public class RollerAgent : Agent
{
    //public NNModel modelSource;
    //Start is called before the first frame update
    public Rigidbody rBody;
    public Transform Target;
    public static Model model_;
    public IWorker worker;
    public float[] FxFz;
    void Start()
    {
        model_ = ModelLoader.LoadFromStreamingAssets("F:\\2Unity\\RollerBall\\Assets\\RollerBallBrain.nn");
        worker = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputePrecompiled, model_);
        rBody = GetComponent<Rigidbody>();
        FxFz = new float[2];
    }

    public override void AgentReset()
    {
        if (this.transform.position.y < 0)
        {
            // If the Agent fell, zero its momentum
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.position = new Vector3(0, 0.5f, 0);
        }

        // Move the target to a new spot
        Target.position = new Vector3(Random.value * 8 - 4,
                                      0.5f,
                                      Random.value * 8 - 4);
    }

    public float speed = 10;
    public void AgentAction(float[] vectorAction)
    {
        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = vectorAction[0];
        controlSignal.z = vectorAction[1];
        rBody.AddForce(controlSignal * speed);

        // Rewards
        float distanceToTarget = Vector3.Distance(this.transform.position,
                                                  Target.position);

        // Reached target
        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            AgentReset();
            Done();
        }

        // User Fell off platform
        if (this.transform.position.y < 0)
        {
            Done();
        }



    }

    // Update is called once per frame
    void Update()
    {

        var input_vector_obs = new Tensor(1, 1, 1, 8, new float[] { Target.position.x, Target.position.y, Target.position.z,
            this.transform.position.x, this.transform.position.y, this.transform.position.z, rBody.velocity.x, rBody.velocity.z });

        var input_eps = new Tensor(1, 1, 1, 1, new float[] { 0.2f });

        var dict = new Dictionary<string, Tensor>();
        dict["vector_observation"] = input_vector_obs;
        dict["epsilon"] = input_eps;

        worker.Execute(dict);
        //worker.Fetch("action_probs").PrintDataPart(24, "action_probs");
        //worker.Fetch("action").PrintDataPart(24, "action:");
        //Debug.Log(worker.Fetch("action").GetType());  //BarracudaToFloatArray
        var out_ = TensorUtils.BarracudaToFloatArray(worker.Fetch("action"));

        int i = 0;
        foreach (var fc in out_)
        {
            FxFz[i] = (float)fc;
            Debug.Log((float)fc);
            i = i + 1;
        }

        AgentAction(FxFz);
        Debug.Log("valueEstimate");
        //UpdateVectorAction(FxFz);
        Debug.Log(GetValueEstimate());
    }
}
