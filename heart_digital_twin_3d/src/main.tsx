import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Deliberately not wrapped in React.StrictMode: its dev-mode double-mount
// behavior conflicts with @react-three/drei's OrbitControls camera
// binding (a known class of issue with three.js side-effect-heavy
// libraries), causing a transient "Cannot read properties of undefined
// (reading 'distanceToSquared')" crash on load.
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
