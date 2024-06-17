# Running the visualization
- Run `python -m visualization.server`
- Visit http://localhost:8000/

# Swapping the pose prediction model
- Edit the `loadModel`, `preprocess`, and `runModel` functions in `visualization/prediction.js`
- If you have an ONNX file, simply place it in the `visualization/models` folder then edit the `MODEL` variable in `visualization/prediction.js` to point to it.
- If you do not want to or are unable to convert the model to ONNX form, `runModel` could point to an inference API.

# Exporting a trained DePOSit model to ONNX
1. Open [export_to_onnx.py](https://github.com/vita-epfl/unposed/tree/main/models/deposit/export_to_onnx.py).
2. Modify `args`, `IN_N`, and `OUT_N` to align with the trained model.
3. Open [deposit.py](https://github.com/vita-epfl/unposed/tree/main/models/deposit/deposit.py).
4. Find the `evaluate` function signature and temporarily change it to `forward` by uncommenting the two lines below it.
5. Run `python -m models.deposit.export_to_onnx`.
	- There may be errors regarding the `opset_version`. Change it as needed in [export_to_onnx.py](https://github.com/vita-epfl/unposed/tree/main/models/deposit/export_to_onnx.py).