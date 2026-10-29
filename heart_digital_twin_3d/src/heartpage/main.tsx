import ReactDOM from "react-dom/client";
import { HeartPage } from "./HeartPage";
import "../cordia/cordia.css";
import "./heart.css";

// Not wrapped in React.StrictMode, for the same OrbitControls reason as src/main.tsx.
ReactDOM.createRoot(document.getElementById("root")!).render(<HeartPage />);
