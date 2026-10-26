/**
 * Which page of the twin this is:
 * - index.html: the clinical twin;
 * - digital.html: the holographic twin -- the same interface, datasets and
 *   simulation, with the heart drawn as a glowing digital model;
 * - workstation.html: the Cordia Heart Twin workstation (patient inputs,
 *   four risk models, trajectory and time machine) around the 3D twin.
 */
export const VARIANT = (document.documentElement.dataset.variant ?? "clinical") as "clinical" | "digital" | "workstation";
/** Pages that show the model hearts (hologram, low-poly, red + blue) and start on them. */
export const DIGITAL = VARIANT !== "clinical";
