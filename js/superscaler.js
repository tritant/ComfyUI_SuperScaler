import { app } from "/scripts/app.js";

// --- Définition des groupes ---
const widgetGroups = {
    "enable_latent_pass": [
        "latent_upscale_by", "latent_denoise", "latent_sampler_name",
        "latent_scheduler", "latent_steps", "latent_cfg"
    ],
    "enable_tiled_pass_2": [ 
        "tiled_upscale_by_2", "tiled_denoise_2", "tile_size_2", "tile_overlap_2",
        "tiled_sampler_name_2", "tiled_scheduler_2", "tiled_steps_2", "tiled_cfg_2"
    ],
    "enable_tiled_pass_3": [
        "tiled_upscale_by_3", "tiled_denoise_3", "tile_size_3", "tile_overlap_3",
        "tiled_sampler_name_3", "tiled_scheduler_3", "tiled_steps_3", "tiled_cfg_3"
    ],
    "enable_sharpen": [
        "sharpen_amount", "sharpen_radius"
    ],
    "enable_grain": [
        "grain_intensity", "grain_type", "grain_size", "saturation_mix", "adaptive_grain"
    ]
};

// Liste de TOUS les widgets gérés (parents et enfants)
const allManagedWidgetNames = new Set();
for (const parentName in widgetGroups) {
    allManagedWidgetNames.add(parentName);
    for (const childName of widgetGroups[parentName]) {
        allManagedWidgetNames.add(childName);
    }
}

function getCleanTitle(toggleName) {
    let title = toggleName.replace("enable_", "").replace(/_/g, " ");
    return title.replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Construit l'interface utilisateur (les widgets) en lisant l'état depuis node.properties.
 * @param {LGraphNode} node - Le nœud ComfyUI
 */
function rebuildWidgets(node) {
    // 1. On efface tous les widgets actuels
    node.widgets.length = 0;

    // 2. On reconstruit la liste en lisant nos propriétés
    for (const toggleName of Object.keys(widgetGroups)) {
        const parentWidget = node.allPythonWidgets.find(w => w.name === toggleName);
        if (!parentWidget) continue;

        // Applique la valeur depuis notre "source de vérité"
        // (Vérifie si la propriété existe, sinon utilise la valeur par défaut du widget)
        parentWidget.value = node.properties.hasOwnProperty(toggleName) 
            ? node.properties[toggleName] 
            : parentWidget.defaultValue;

        // --- AJOUT DU SÉPARATEUR ---
        node.widgets.push({ 
            name: "separator_spacer_" + parentWidget.name,
            type: "CUSTOM_SPACER", 
            draw: (ctx, node, width, y) => {
                const title = getCleanTitle(parentWidget.name);
                const rectHeight = 20, marginY = 5, x_padding = 10;
                ctx.fillStyle = "#272";
                ctx.fillRect(x_padding, y + marginY, width - (x_padding * 2), rectHeight);
                ctx.fillStyle = "#CCC";
                ctx.font = "bold 12px Arial";
                ctx.textAlign = "center";
                const textY = y + marginY + rectHeight / 2 + 4;
                ctx.fillText(title, width / 2, textY); 
            }, 
            computeSize: () => [0, 30]
        });

        // A. On ajoute le "parent" (toggle)
        node.widgets.push(parentWidget);
        
        // B. On vérifie (en lisant nos propriétés) s'il faut ajouter ses enfants
        const showChildren = parentWidget.value;
        
        if (showChildren) {
            const childrenNames = widgetGroups[toggleName];
            for (const childName of childrenNames) {
                const childWidget = node.allPythonWidgets.find(w => w.name === childName);
                if (childWidget) {
                    // Applique la valeur depuis notre "source de vérité"
                    childWidget.value = node.properties.hasOwnProperty(childName)
                        ? node.properties[childName]
                        : childWidget.defaultValue;
                    node.widgets.push(childWidget);
                } else {
                    console.warn(`[SuperScaler] Enfant widget '${childName}' NON TROUVÉ.`);
                }
            }
        }
    }
    
    // --- AJOUT : AFFICHER LA SEED GLOBALE ---
    const globalSeedWidget = node.allPythonWidgets.find(w => w.name === "seed");
    
    if (globalSeedWidget && !allManagedWidgetNames.has("seed")) {
        // Ajouter le spacer pour la seed
        node.widgets.push({ 
            name: "separator_spacer_global_seed",
            type: "CUSTOM_SPACER", 
            draw: (ctx, node, width, y) => {
                const title = "Global Seed";
                const rectHeight = 20, marginY = 5, x_padding = 10;
                ctx.fillStyle = "#272";
                ctx.fillRect(x_padding, y + marginY, width - (x_padding * 2), rectHeight);
                ctx.fillStyle = "#CCC";
                ctx.font = "bold 12px Arial";
                ctx.textAlign = "center";
                const textY = y + marginY + rectHeight / 2 + 4;
                ctx.fillText(title, width / 2, textY); 
            }, 
            computeSize: () => [0, 30]
        });
    }

    // On ajoute tous les widgets qui n'étaient pas gérés
    // (Cela inclut la seed ET son menu compagnon)
    for (const widget of node.allPythonWidgets) {
        if (!allManagedWidgetNames.has(widget.name)) {
            // Applique la valeur depuis notre "source de vérité"
            if (node.properties.hasOwnProperty(widget.name)) {
                widget.value = node.properties[widget.name];
            }
            node.widgets.push(widget);
        }
    }
    // --- FIN DE L'AJOUT ---
    
    // 3. Force le redessinage
    const newComputedSize = node.computeSize();
    node.size[1] = newComputedSize[1];
    
    if (app.graph) {
        app.graph.setDirtyCanvas(true, true);
    }
}


app.registerExtension({
    name: "SuperScaler.DynamicWidgets.vSafe",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        if (nodeData.name === "SuperScaler_Pipeline") {
            
            // --- C'est ici que la logique de ton exemple 'Orchestrator' est appliquée ---
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                
                const node = this;
                
                // 1. Initialise la "source de vérité" des propriétés (comme Orchestrator)
                if (!node.properties) {
                    node.properties = {};
                }
                
                // 2. Sauvegarde les widgets "templates" (avec leurs valeurs par défaut)
                node.allPythonWidgets = [...node.widgets];
                
                // 3. Remplit la "source de vérité" avec les valeurs par défaut (par NOM)
                //    UNIQUEMENT si elles n'existent pas déjà (chargées par le workflow)
                for (const widget of node.allPythonWidgets) {
                    if (widget.name && !node.properties.hasOwnProperty(widget.name)) {
                        node.properties[widget.name] = widget.value;
                    }
                    
                    // Stocke la valeur par défaut pour la réinitialisation
                    widget.defaultValue = widget.value;
                }
                
                // 4. Attache les callbacks (une seule fois)
                // Le callback met à jour 'properties' et reconstruit l'UI
                if (!node.callbacksAttached) {
                    for (const widget of node.allPythonWidgets) {
                        if (widget.callback) {
                            const originalCallback = widget.callback;
                            widget.callback = (value, ...args) => {
                                // Exécute le callback original (s'il existe)
                                originalCallback?.call(widget, value, ...args);
                                
                                // Met à jour la "source de vérité"
                                node.properties[widget.name] = value;
                                
                                // Reconstruit l'interface
                                rebuildWidgets(node); 
                            };
                        }
                    }
                    node.callbacksAttached = true; // On pose le drapeau
                }

                // Lance la construction initiale de l'UI
                // (utilise un court délai, comme ton 'Orchestrator', pour s'assurer que tout est chargé)
                setTimeout(() => rebuildWidgets(node), 10);
            };

            // Garde une trace de la fonction 'onConfigure' originale
            // Nous l'utilisons SEULEMENT pour fusionner les valeurs chargées
            const originalOnConfigure = nodeType.prototype.onConfigure;

            nodeType.prototype.onConfigure = function(values) {
                originalOnConfigure?.apply(this, arguments);
                
                // Quand ComfyUI charge un workflow, il met les valeurs dans this.properties
                // (remplaçant 'values' qui est déprécié)
                // Nous devons juste nous assurer que notre 'rebuild' est appelé après.
                if (this.properties) {
                     // 'values' (l'argument) est souvent l'ancien 'properties'
                    Object.assign(this.properties, values);
                }

                // Force une reconstruction de l'UI au cas où
                if (this.allPythonWidgets) {
                    rebuildWidgets(this);
                }
            };
        }
    }
});