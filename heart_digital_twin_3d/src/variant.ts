/**
 * Which page of the twin this is. index.html is the clinical twin;
 * digital.html is the holographic twin -- the same interface, datasets and
 * simulation, with the heart drawn as a glowing digital model.
 */
export const DIGITAL = document.documentElement.dataset.variant === "digital";
