import ReactDOM from "react-dom/client";
import { useWs } from "../workstation/wsStore";
import { CordiaApp } from "./CordiaApp";
import "./cordia.css";

// The dashboard opens on the example patient at a healthy resting rate and pressure.
const start = { ...useWs.getState().S, heart_rate: 72, sys_bp: 120, dia_bp: 80 };
useWs.setState({ base: start, S: start });

// Not wrapped in React.StrictMode, for the same OrbitControls reason as src/main.tsx.
ReactDOM.createRoot(document.getElementById("root")!).render(<CordiaApp />);
