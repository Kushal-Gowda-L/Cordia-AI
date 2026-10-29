import ReactDOM from "react-dom/client";
import { Workstation } from "./Workstation";
import "./workstation.css";

// Not wrapped in React.StrictMode, for the same OrbitControls reason as src/main.tsx.
ReactDOM.createRoot(document.getElementById("root")!).render(<Workstation />);
